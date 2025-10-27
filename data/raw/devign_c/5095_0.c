static void SET_TYPE(resample_one)(ResampleContext *c,

                                   void *dst0, int dst_index, const void *src0,

                                   int src_size, int index, int frac)

{

    FELEM *dst = dst0;

    const FELEM *src = src0;

    int i;

    int sample_index = index >> c->phase_shift;

    FELEM2 val = 0;

    FELEM *filter = ((FELEM *)c->filter_bank) +

                    c->filter_length * (index & c->phase_mask);



    if (sample_index < 0) {

        for (i = 0; i < c->filter_length; i++)

            val += src[FFABS(sample_index + i) % src_size] *

                   (FELEM2)filter[i];

    } else if (c->linear) {

        FELEM2 v2 = 0;

        for (i = 0; i < c->filter_length; i++) {

            val += src[abs(sample_index + i)] * (FELEM2)filter[i];

            v2  += src[abs(sample_index + i)] * (FELEM2)filter[i + c->filter_length];

        }

        val += (v2 - val) * (FELEML)frac / c->src_incr;

    } else {

        for (i = 0; i < c->filter_length; i++)

            val += src[sample_index + i] * (FELEM2)filter[i];

    }



    OUT(dst[dst_index], val);

}
