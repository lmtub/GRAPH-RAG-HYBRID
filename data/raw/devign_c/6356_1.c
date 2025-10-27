static void do_adaptive_prediction(struct G722Band *band, const int cur_diff)

{

    int sg[2], limit, cur_qtzd_reconst;



    const int cur_part_reconst = band->s_zero + cur_diff < 0;



    sg[0] = sign_lookup[cur_part_reconst != band->part_reconst_mem[0]];

    sg[1] = sign_lookup[cur_part_reconst == band->part_reconst_mem[1]];

    band->part_reconst_mem[1] = band->part_reconst_mem[0];

    band->part_reconst_mem[0] = cur_part_reconst;



    band->pole_mem[1] = av_clip((sg[0] * av_clip(band->pole_mem[0], -8191, 8191) >> 5) +

                                (sg[1] << 7) + (band->pole_mem[1] * 127 >> 7), -12288, 12288);



    limit = 15360 - band->pole_mem[1];

    band->pole_mem[0] = av_clip(-192 * sg[0] + (band->pole_mem[0] * 255 >> 8), -limit, limit);



    s_zero(cur_diff, band);



    cur_qtzd_reconst = av_clip_int16((band->s_predictor + cur_diff) << 1);

    band->s_predictor = av_clip_int16(band->s_zero +

                                      (band->pole_mem[0] * cur_qtzd_reconst >> 15) +

                                      (band->pole_mem[1] * band->prev_qtzd_reconst >> 15));

    band->prev_qtzd_reconst = cur_qtzd_reconst;

}
