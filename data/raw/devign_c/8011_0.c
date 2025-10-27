static void omap_ppm_save(const char *filename, uint8_t *data,

                    int w, int h, int linesize, Error **errp)

{

    FILE *f;

    uint8_t *d, *d1;

    unsigned int v;

    int ret, y, x, bpp;



    f = fopen(filename, "wb");

    if (!f) {

        error_setg(errp, "failed to open file '%s': %s", filename,

                   strerror(errno));

        return;

    }

    ret = fprintf(f, "P6\n%d %d\n%d\n", w, h, 255);

    if (ret < 0) {

        goto write_err;

    }

    d1 = data;

    bpp = linesize / w;

    for (y = 0; y < h; y ++) {

        d = d1;

        for (x = 0; x < w; x ++) {

            v = *(uint32_t *) d;

            switch (bpp) {

            case 2:

                ret = fputc((v >> 8) & 0xf8, f);

                if (ret == EOF) {

                    goto write_err;

                }

                ret = fputc((v >> 3) & 0xfc, f);

                if (ret == EOF) {

                    goto write_err;

                }

                ret = fputc((v << 3) & 0xf8, f);

                if (ret == EOF) {

                    goto write_err;

                }

                break;

            case 3:

            case 4:

            default:

                ret = fputc((v >> 16) & 0xff, f);

                if (ret == EOF) {

                    goto write_err;

                }

                ret = fputc((v >> 8) & 0xff, f);

                if (ret == EOF) {

                    goto write_err;

                }

                ret = fputc((v) & 0xff, f);

                if (ret == EOF) {

                    goto write_err;

                }

                break;

            }

            d += bpp;

        }

        d1 += linesize;

    }

out:

    fclose(f);

    return;



write_err:

    error_setg(errp, "failed to write to file '%s': %s", filename,

               strerror(errno));

    unlink(filename);

    goto out;

}
