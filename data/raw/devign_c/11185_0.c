int qemu_pixman_get_type(int rshift, int gshift, int bshift)

{

    int type = PIXMAN_TYPE_OTHER;



    if (rshift > gshift && gshift > bshift) {

        if (bshift == 0) {

            type = PIXMAN_TYPE_ARGB;

        } else {

#if PIXMAN_VERSION >= PIXMAN_VERSION_ENCODE(0, 21, 8)

            type = PIXMAN_TYPE_RGBA;

#endif

        }

    } else if (rshift < gshift && gshift < bshift) {

        if (rshift == 0) {

            type = PIXMAN_TYPE_ABGR;

        } else {

#if PIXMAN_VERSION >= PIXMAN_VERSION_ENCODE(0, 21, 8)

            type = PIXMAN_TYPE_BGRA;

#endif

        }

    }

    return type;

}
