#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libfreenect.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: kinect_tilt <deg> [-30..30]\n"); return 1; }
    double angle = atof(argv[1]);
    if (angle < -30 || angle > 30) { fprintf(stderr, "angle out of range\n"); return 1; }

    freenect_context *ctx;
    freenect_device *dev;
    if (freenect_init(&ctx, NULL) < 0) return 1;
    freenect_select_subdevices(ctx, (freenect_device_flags)(FREENECT_DEVICE_CAMERA | FREENECT_DEVICE_MOTOR));
    if (freenect_num_devices(ctx) < 1) { fprintf(stderr, "no motor\n"); return 1; }
    if (freenect_open_device(ctx, &dev, 0) < 0) { fprintf(stderr, "open fail\n"); return 1; }

    if (freenect_set_tilt_degs(dev, angle) < 0) { fprintf(stderr, "tilt fail\n"); return 1; }
    int ms = (argc > 2) ? atoi(argv[2]) : 1500;  // wait for move
    freenect_set_led(dev, LED_GREEN);
    sleep(ms / 1000 + 1);
    freenect_raw_tilt_state *st = freenect_get_tilt_state(dev);
    freenect_update_tilt_state(dev);
    st = freenect_get_tilt_state(dev);
    printf("tilt now: %.1f deg\n", freenect_get_tilt_degs(st));

    freenect_set_led(dev, LED_OFF);
    freenect_close_device(dev);
    freenect_shutdown(ctx);
    return 0;
}
