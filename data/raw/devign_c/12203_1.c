static BlockDriver *bdrv_probe_all(const uint8_t *buf, int buf_size,

                                   const char *filename)

{

    int score_max = 0, score;

    BlockDriver *drv = NULL, *d;



    QLIST_FOREACH(d, &bdrv_drivers, list) {

        if (d->bdrv_probe) {

            score = d->bdrv_probe(buf, buf_size, filename);

            if (score > score_max) {

                score_max = score;

                drv = d;

            }

        }

    }



    return drv;

}
