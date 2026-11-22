/*
 * kinect_hub -- single owner of every Kinect 360 (1473) subdevice.
 *
 * On the 1473 the tilt motor is driven THROUGH the audio device, so camera,
 * microphones and motor must live in one libfreenect context. This daemon
 * opens all three and exposes them to other processes:
 *
 *   stdout : raw audio, 16 kHz, 4 channels interleaved, int16 little endian
 *   /dev/shm/kinect_hub : header + latest depth_mm (640x480 u16) + latest RGB
 *   stdin  : text commands, one per line
 *              tilt <deg>        move motor (-27..27)
 *              led <0..6>        set LED
 *              mode depth|rgb|both   choose which camera stream runs
 *              quit
 *   stderr : log lines
 *
 * Build: gcc -O2 -I$HOME/.kinect/include/libfreenect kinect_hub.c -o kinect_hub \
 *        -L$HOME/.kinect/lib -Wl,-rpath,$HOME/.kinect/lib -lfreenect -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <libfreenect.h>
#include <libfreenect_audio.h>

#define W 640
#define H 480
#define SHM_NAME "/kinect_hub"
#define HUB_MAGIC 0x4b48554a /* 'JUHK' */

enum { MODE_DEPTH = 1, MODE_RGB = 2, MODE_BOTH = 3 };

/* Seqlock convention: writer makes seq odd, writes, makes it even. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    volatile uint32_t depth_seq;
    volatile uint32_t rgb_seq;
    volatile uint32_t state_seq;
    uint32_t mode;
    double depth_ts;
    double rgb_ts;
    double state_ts;
    double tilt_deg;      /* measured angle reported by the device */
    double tilt_target;   /* last commanded angle */
    double accel[3];      /* m/s^2 */
    int32_t tilt_status;  /* 0 stopped, 1 limit, 4 moving */
    int32_t audio_rate;
    double last_move_ts;  /* wall time of last tilt command */
    uint8_t pad[64];
} hub_header;

#define DEPTH_OFF 4096
#define RGB_OFF (DEPTH_OFF + W * H * 2)
#define SHM_SIZE (RGB_OFF + W * H * 3)

static volatile sig_atomic_t running = 1;
static uint8_t *shm;
static hub_header *hdr;
static int cur_mode = 0;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void on_sig(int s) { (void)s; running = 0; }

static void depth_cb(freenect_device *dev, void *data, uint32_t ts) {
    (void)dev; (void)ts;
    __atomic_add_fetch(&hdr->depth_seq, 1, __ATOMIC_SEQ_CST);
    memcpy(shm + DEPTH_OFF, data, W * H * 2);
    hdr->depth_ts = now_s();
    __atomic_add_fetch(&hdr->depth_seq, 1, __ATOMIC_SEQ_CST);
}

static void video_cb(freenect_device *dev, void *data, uint32_t ts) {
    (void)dev; (void)ts;
    __atomic_add_fetch(&hdr->rgb_seq, 1, __ATOMIC_SEQ_CST);
    memcpy(shm + RGB_OFF, data, W * H * 3);
    hdr->rgb_ts = now_s();
    __atomic_add_fetch(&hdr->rgb_seq, 1, __ATOMIC_SEQ_CST);
}

static int16_t audio_buf[4 * 4096];

static void audio_cb(freenect_device *dev, int n, int32_t *m1, int32_t *m2,
                     int32_t *m3, int32_t *m4, int16_t *cancelled, void *unknown) {
    (void)dev; (void)cancelled; (void)unknown;
    while (n > 0) {
        int chunk = n > 4096 ? 4096 : n;
        for (int i = 0; i < chunk; i++) {
            audio_buf[4 * i + 0] = (int16_t)(m1[i] >> 16);
            audio_buf[4 * i + 1] = (int16_t)(m2[i] >> 16);
            audio_buf[4 * i + 2] = (int16_t)(m3[i] >> 16);
            audio_buf[4 * i + 3] = (int16_t)(m4[i] >> 16);
        }
        if (fwrite(audio_buf, sizeof(int16_t) * 4, chunk, stdout) != (size_t)chunk) {
            running = 0;
            return;
        }
        m1 += chunk; m2 += chunk; m3 += chunk; m4 += chunk;
        n -= chunk;
    }
    fflush(stdout);
}

static void set_mode(freenect_device *dev, int mode) {
    if (mode == cur_mode) return;
    if ((cur_mode & MODE_DEPTH) && !(mode & MODE_DEPTH)) freenect_stop_depth(dev);
    if ((cur_mode & MODE_RGB) && !(mode & MODE_RGB)) freenect_stop_video(dev);
    if (!(cur_mode & MODE_DEPTH) && (mode & MODE_DEPTH)) {
        if (freenect_start_depth(dev) < 0) fprintf(stderr, "hub: start_depth failed\n");
    }
    if (!(cur_mode & MODE_RGB) && (mode & MODE_RGB)) {
        if (freenect_start_video(dev) < 0) fprintf(stderr, "hub: start_video failed\n");
    }
    cur_mode = mode;
    hdr->mode = (uint32_t)mode;
    fprintf(stderr, "hub: mode=%d\n", mode);
}

static void handle_cmd(freenect_device *dev, char *line) {
    char word[32] = {0}, arg[32] = {0};
    if (sscanf(line, "%31s %31s", word, arg) < 1) return;
    if (!strcmp(word, "tilt")) {
        double a = atof(arg);
        if (a < -27) a = -27;
        if (a > 27) a = 27;
        hdr->tilt_target = a;
        hdr->last_move_ts = now_s();
        if (freenect_set_tilt_degs(dev, a) < 0) fprintf(stderr, "hub: tilt failed\n");
    } else if (!strcmp(word, "led")) {
        freenect_set_led(dev, (freenect_led_options)atoi(arg));
    } else if (!strcmp(word, "mode")) {
        if (!strcmp(arg, "depth")) set_mode(dev, MODE_DEPTH);
        else if (!strcmp(arg, "rgb")) set_mode(dev, MODE_RGB);
        else if (!strcmp(arg, "both")) set_mode(dev, MODE_BOTH);
    } else if (!strcmp(word, "quit")) {
        running = 0;
    }
}

static void update_state(freenect_device *dev) {
    if (freenect_update_tilt_state(dev) < 0) return;
    freenect_raw_tilt_state *st = freenect_get_tilt_state(dev);
    double ax, ay, az;
    freenect_get_mks_accel(st, &ax, &ay, &az);
    __atomic_add_fetch(&hdr->state_seq, 1, __ATOMIC_SEQ_CST);
    hdr->tilt_deg = freenect_get_tilt_degs(st);
    hdr->accel[0] = ax; hdr->accel[1] = ay; hdr->accel[2] = az;
    hdr->tilt_status = (int32_t)freenect_get_tilt_status(st);
    hdr->state_ts = now_s();
    __atomic_add_fetch(&hdr->state_seq, 1, __ATOMIC_SEQ_CST);
}

int main(int argc, char **argv) {
    int want_audio = 1;
    int start_mode = MODE_DEPTH;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-audio")) want_audio = 0;
        else if (!strcmp(argv[i], "--both")) start_mode = MODE_BOTH;
    }
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    signal(SIGPIPE, on_sig);

    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0600);
    if (fd < 0 || ftruncate(fd, SHM_SIZE) < 0) { perror("shm"); return 1; }
    shm = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (shm == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);
    memset(shm, 0, sizeof(hub_header));
    hdr = (hub_header *)shm;
    hdr->magic = HUB_MAGIC;
    hdr->version = 1;
    hdr->audio_rate = 16000;

    freenect_context *ctx;
    freenect_device *dev;
    if (freenect_init(&ctx, NULL) < 0) { fprintf(stderr, "hub: init failed\n"); return 1; }
    freenect_set_log_level(ctx, FREENECT_LOG_WARNING);
    int flags = FREENECT_DEVICE_CAMERA | FREENECT_DEVICE_MOTOR;
    if (want_audio) flags |= FREENECT_DEVICE_AUDIO;
    freenect_select_subdevices(ctx, (freenect_device_flags)flags);
    if (freenect_num_devices(ctx) < 1) { fprintf(stderr, "hub: no kinect\n"); return 1; }
    if (freenect_open_device(ctx, &dev, 0) < 0) { fprintf(stderr, "hub: open failed\n"); return 1; }

    freenect_set_depth_callback(dev, depth_cb);
    freenect_set_video_callback(dev, video_cb);
    freenect_set_depth_mode(dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM));
    freenect_set_video_mode(dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
    set_mode(dev, start_mode);

    if (want_audio) {
        freenect_set_audio_in_callback(dev, audio_cb);
        if (freenect_start_audio(dev) < 0) fprintf(stderr, "hub: start_audio failed\n");
    }
    freenect_set_led(dev, LED_GREEN);
    fprintf(stderr, "hub: ready (audio=%d)\n", want_audio);

    int fl = fcntl(STDIN_FILENO, F_GETFL);
    fcntl(STDIN_FILENO, F_SETFL, fl | O_NONBLOCK);
    char line[256];
    size_t llen = 0;
    double last_state = 0;
    struct timeval tv = {0, 20000};

    while (running) {
        if (freenect_process_events_timeout(ctx, &tv) < 0) {
            fprintf(stderr, "hub: process_events error\n");
            break;
        }
        struct pollfd p = {STDIN_FILENO, POLLIN, 0};
        if (poll(&p, 1, 0) > 0) {
            char c;
            ssize_t r;
            while ((r = read(STDIN_FILENO, &c, 1)) == 1) {
                if (c == '\n' || llen == sizeof(line) - 1) {
                    line[llen] = 0;
                    handle_cmd(dev, line);
                    llen = 0;
                } else {
                    line[llen++] = c;
                }
            }
            if (r == 0) running = 0; /* parent closed stdin */
        }
        double t = now_s();
        if (t - last_state > 0.1) { update_state(dev); last_state = t; }
    }

    fprintf(stderr, "hub: shutting down\n");
    if (want_audio) freenect_stop_audio(dev);
    set_mode(dev, 0);
    freenect_set_led(dev, LED_OFF);
    freenect_close_device(dev);
    freenect_shutdown(ctx);
    munmap(shm, SHM_SIZE);
    shm_unlink(SHM_NAME);
    return 0;
}
