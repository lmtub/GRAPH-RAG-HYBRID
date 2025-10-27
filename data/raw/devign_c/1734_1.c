static void quantize_and_encode_band_mips(struct AACEncContext *s, PutBitContext *pb,

                                          const float *in, float *out, int size, int scale_idx,

                                          int cb, const float lambda, int rtz)

{

    quantize_and_encode_band_cost(s, pb, in, out, NULL, size, scale_idx, cb, lambda,

                                  INFINITY, NULL, (rtz) ? ROUND_TO_ZERO : ROUND_STANDARD);

}
