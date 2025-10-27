static int parse_cube(AVFilterContext *ctx, FILE *f)

{

    LUT3DContext *lut3d = ctx->priv;

    char line[MAX_LINE_SIZE];

    float min[3] = {0.0, 0.0, 0.0};

    float max[3] = {1.0, 1.0, 1.0};



    while (fgets(line, sizeof(line), f)) {

        if (!strncmp(line, "LUT_3D_SIZE ", 12)) {

            int i, j, k;

            const int size = strtol(line + 12, NULL, 0);



            if (size > MAX_LEVEL) {

                av_log(ctx, AV_LOG_ERROR, "Too large 3D LUT\n");

                return AVERROR(EINVAL);

            }

            lut3d->lutsize = size;

            for (k = 0; k < size; k++) {

                for (j = 0; j < size; j++) {

                    for (i = 0; i < size; i++) {

                        struct rgbvec *vec = &lut3d->lut[k][j][i];



                        do {

                            NEXT_LINE(0);

                            if (!strncmp(line, "DOMAIN_", 7)) {

                                float *vals = NULL;

                                if      (!strncmp(line + 7, "MIN ", 4)) vals = min;

                                else if (!strncmp(line + 7, "MAX ", 4)) vals = max;

                                if (!vals)

                                    return AVERROR_INVALIDDATA;

                                sscanf(line + 11, "%f %f %f", vals, vals + 1, vals + 2);

                                av_log(ctx, AV_LOG_DEBUG, "min: %f %f %f | max: %f %f %f\n",

                                       min[0], min[1], min[2], max[0], max[1], max[2]);

                                continue;

                            }

                        } while (skip_line(line));

                        if (sscanf(line, "%f %f %f", &vec->r, &vec->g, &vec->b) != 3)

                            return AVERROR_INVALIDDATA;

                        vec->r *= max[0] - min[0];

                        vec->g *= max[1] - min[1];

                        vec->b *= max[2] - min[2];

                    }

                }

            }

            break;

        }

    }

    return 0;

}
