void helper_set_alt_mode (void)

{

    env->saved_mode = env->ps & 0xC;

    env->ps = (env->ps & ~0xC) | (env->ipr[IPR_ALT_MODE] & 0xC);

}
