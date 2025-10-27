static void vnc_convert_pixel(VncState *vs, uint8_t *buf, uint32_t v)

{

    uint8_t r, g, b;



    r = ((v >> vs->server_red_shift) & vs->server_red_max) * (vs->client_red_max + 1) /

        (vs->server_red_max + 1);

    g = ((v >> vs->server_green_shift) & vs->server_green_max) * (vs->client_green_max + 1) /

        (vs->server_green_max + 1);

    b = ((v >> vs->server_blue_shift) & vs->server_blue_max) * (vs->client_blue_max + 1) /

        (vs->server_blue_max + 1);

    v = (r << vs->client_red_shift) |

        (g << vs->client_green_shift) |

        (b << vs->client_blue_shift);

    switch(vs->pix_bpp) {

    case 1:

        buf[0] = v;

        break;

    case 2:

        if (vs->pix_big_endian) {

            buf[0] = v >> 8;

            buf[1] = v;

        } else {

            buf[1] = v >> 8;

            buf[0] = v;

        }

        break;

    default:

    case 4:

        if (vs->pix_big_endian) {

            buf[0] = v >> 24;

            buf[1] = v >> 16;

            buf[2] = v >> 8;

            buf[3] = v;

        } else {

            buf[3] = v >> 24;

            buf[2] = v >> 16;

            buf[1] = v >> 8;

            buf[0] = v;

        }

        break;

    }

}
