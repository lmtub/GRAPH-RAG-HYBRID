static void apply_mid_side_stereo(ChannelElement *cpe)

{

    int w, w2, g, i;

    IndividualChannelStream *ics = &cpe->ch[0].ics;

    if (!cpe->common_window)

        return;

    for (w = 0; w < ics->num_windows; w += ics->group_len[w]) {

        for (w2 =  0; w2 < ics->group_len[w]; w2++) {

            int start = (w+w2) * 128;

            for (g = 0; g < ics->num_swb; g++) {

                if (!cpe->ms_mask[w*16 + g]) {

                    start += ics->swb_sizes[g];

                    continue;

                }

                for (i = 0; i < ics->swb_sizes[g]; i++) {

                    float L = (cpe->ch[0].coeffs[start+i] + cpe->ch[1].coeffs[start+i]) * 0.5f;

                    float R = L - cpe->ch[1].coeffs[start+i];

                    cpe->ch[0].coeffs[start+i] = L;

                    cpe->ch[1].coeffs[start+i] = R;

                }

                start += ics->swb_sizes[g];

            }

        }

    }

}
