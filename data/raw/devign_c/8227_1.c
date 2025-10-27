static void apply_independent_coupling_fixed(AACContext *ac,

                                       SingleChannelElement *target,

                                       ChannelElement *cce, int index)

{

    int i, c, shift, round, tmp;

    const int gain = cce->coup.gain[index][0];

    const int *src = cce->ch[0].ret;

    int *dest = target->ret;

    const int len = 1024 << (ac->oc[1].m4ac.sbr == 1);



    c = cce_scale_fixed[gain & 7];

    shift = (gain-1024) >> 3;

    if (shift < 0) {

        shift = -shift;

        round = 1 << (shift - 1);



        for (i = 0; i < len; i++) {

            tmp = (int)(((int64_t)src[i] * c + (int64_t)0x1000000000) >> 37);

            dest[i] += (tmp + round) >> shift;

        }

    }

    else {

      for (i = 0; i < len; i++) {

          tmp = (int)(((int64_t)src[i] * c + (int64_t)0x1000000000) >> 37);

          dest[i] += tmp << shift;

      }

    }

}
