void ff_jpegls_init_state(JLSState *state){

    int i;



    state->twonear = state->near * 2 + 1;

    state->range = ((state->maxval + state->twonear - 1) / state->twonear) + 1;



    // QBPP = ceil(log2(RANGE))

    for(state->qbpp = 0; (1 << state->qbpp) < state->range; state->qbpp++);



    if(state->bpp < 8)

        state->limit = 16 + 2 * state->bpp - state->qbpp;

    else

        state->limit = (4 * state->bpp) - state->qbpp;



    for(i = 0; i < 367; i++) {

        state->A[i] = FFMAX((state->range + 32) >> 6, 2);

        state->N[i] = 1;

    }



}
