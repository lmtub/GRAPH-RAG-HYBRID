static void copy_frame(Jpeg2000EncoderContext *s)

{

    int tileno, compno, i, y, x;

    uint8_t *line;

    for (tileno = 0; tileno < s->numXtiles * s->numYtiles; tileno++){

        Jpeg2000Tile *tile = s->tile + tileno;

        if (s->planar){

            for (compno = 0; compno < s->ncomponents; compno++){

                Jpeg2000Component *comp = tile->comp + compno;

                int *dst = comp->data;

                line = s->picture.data[compno]

                       + comp->coord[1][0] * s->picture.linesize[compno]

                       + comp->coord[0][0];

                for (y = comp->coord[1][0]; y < comp->coord[1][1]; y++){

                    uint8_t *ptr = line;

                    for (x = comp->coord[0][0]; x < comp->coord[0][1]; x++)

                        *dst++ = *ptr++ - (1 << 7);

                    line += s->picture.linesize[compno];

                }

            }

        } else{

            line = s->picture.data[0] + tile->comp[0].coord[1][0] * s->picture.linesize[0]

                   + tile->comp[0].coord[0][0] * s->ncomponents;



            i = 0;

            for (y = tile->comp[0].coord[1][0]; y < tile->comp[0].coord[1][1]; y++){

                uint8_t *ptr = line;

                for (x = tile->comp[0].coord[0][0]; x < tile->comp[0].coord[0][1]; x++, i++){

                    for (compno = 0; compno < s->ncomponents; compno++){

                        tile->comp[compno].data[i] = *ptr++  - (1 << 7);

                    }

                }

                line += s->picture.linesize[0];

            }

        }

    }

}
