static void find_block_motion(DeshakeContext *deshake, uint8_t *src1,

                              uint8_t *src2, int cx, int cy, int stride,

                              MotionVector *mv)

{

    int x, y;

    int diff;

    int smallest = INT_MAX;

    int tmp, tmp2;



    #define CMP(i, j) deshake->c.sad[0](deshake, src1 + cy * stride + cx, \

                                        src2 + (j) * stride + (i), stride, \

                                        deshake->blocksize)



    if (deshake->search == EXHAUSTIVE) {

        // Compare every possible position - this is sloooow!

        for (y = -deshake->ry; y <= deshake->ry; y++) {

            for (x = -deshake->rx; x <= deshake->rx; x++) {

                diff = CMP(cx - x, cy - y);

                if (diff < smallest) {

                    smallest = diff;

                    mv->x = x;

                    mv->y = y;

                }

            }

        }

    } else if (deshake->search == SMART_EXHAUSTIVE) {

        // Compare every other possible position and find the best match

        for (y = -deshake->ry + 1; y < deshake->ry - 2; y += 2) {

            for (x = -deshake->rx + 1; x < deshake->rx - 2; x += 2) {

                diff = CMP(cx - x, cy - y);

                if (diff < smallest) {

                    smallest = diff;

                    mv->x = x;

                    mv->y = y;

                }

            }

        }



        // Hone in on the specific best match around the match we found above

        tmp = mv->x;

        tmp2 = mv->y;



        for (y = tmp2 - 1; y <= tmp2 + 1; y++) {

            for (x = tmp - 1; x <= tmp + 1; x++) {

                if (x == tmp && y == tmp2)

                    continue;



                diff = CMP(cx - x, cy - y);

                if (diff < smallest) {

                    smallest = diff;

                    mv->x = x;

                    mv->y = y;

                }

            }

        }

    }



    if (smallest > 512) {

        mv->x = -1;

        mv->y = -1;

    }

    emms_c();

    //av_log(NULL, AV_LOG_ERROR, "%d\n", smallest);

    //av_log(NULL, AV_LOG_ERROR, "Final: (%d, %d) = %d x %d\n", cx, cy, mv->x, mv->y);

}
