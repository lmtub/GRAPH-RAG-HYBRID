static int mux_chr_write(CharDriverState *chr, const uint8_t *buf, int len)

{

    MuxDriver *d = chr->opaque;

    int ret;

    if (!d->timestamps) {

        ret = d->drv->chr_write(d->drv, buf, len);

    } else {

        int i;



        ret = 0;

        for (i = 0; i < len; i++) {

            if (d->linestart) {

                char buf1[64];

                int64_t ti;

                int secs;



                ti = qemu_get_clock(rt_clock);

                if (d->timestamps_start == -1)

                    d->timestamps_start = ti;

                ti -= d->timestamps_start;

                secs = ti / 1000;

                snprintf(buf1, sizeof(buf1),

                         "[%02d:%02d:%02d.%03d] ",

                         secs / 3600,

                         (secs / 60) % 60,

                         secs % 60,

                         (int)(ti % 1000));

                d->drv->chr_write(d->drv, (uint8_t *)buf1, strlen(buf1));

                d->linestart = 0;

            }

            ret += d->drv->chr_write(d->drv, buf+i, 1);

            if (buf[i] == '\n') {

                d->linestart = 1;

            }

        }

    }

    return ret;

}
