void qemu_pixman_linebuf_fill(pixman_image_t *linebuf, pixman_image_t *fb,

                              int width, int y)

{

    pixman_image_composite(PIXMAN_OP_SRC, fb, NULL, linebuf,

                           0, y, 0, 0, 0, 0, width, 1);

}
