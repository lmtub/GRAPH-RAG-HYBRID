pixman_format_code_t qemu_default_pixman_format(int bpp, bool native_endian)

{

    if (native_endian) {

        switch (bpp) {

        case 15:

            return PIXMAN_x1r5g5b5;

        case 16:

            return PIXMAN_r5g6b5;

        case 24:

            return PIXMAN_r8g8b8;

        case 32:

            return PIXMAN_x8r8g8b8;

        }

    } else {

        switch (bpp) {

        case 24:

            return PIXMAN_b8g8r8;

        case 32:

            return PIXMAN_b8g8r8x8;

        break;

        }

    }

    g_assert_not_reached();

}
