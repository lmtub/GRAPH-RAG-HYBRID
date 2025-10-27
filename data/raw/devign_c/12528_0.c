static void raw_reopen_abort(BDRVReopenState *state)

{

    g_free(state->opaque);

    state->opaque = NULL;

}
