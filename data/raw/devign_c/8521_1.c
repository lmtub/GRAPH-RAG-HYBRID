static int interp(RA144Context *ractx, int16_t *out, int a,

                  int copyold, int energy)

{

    int work[10];

    int b = NBLOCKS - a;

    int i;



    // Interpolate block coefficients from the this frame's forth block and

    // last frame's forth block.

    for (i=0; i<30; i++)

        out[i] = (a * ractx->lpc_coef[0][i] + b * ractx->lpc_coef[1][i])>> 2;



    if (eval_refl(work, out, ractx)) {

        // The interpolated coefficients are unstable, copy either new or old

        // coefficients.

        int_to_int16(out, ractx->lpc_coef[copyold]);

        return rescale_rms(ractx->lpc_refl_rms[copyold], energy);

    } else {

        return rescale_rms(rms(work), energy);

    }

}
