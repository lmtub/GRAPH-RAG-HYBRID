void kqemu_flush(CPUState *env, int global)

{

    LOG_INT("kqemu_flush:\n");

    nb_pages_to_flush = KQEMU_FLUSH_ALL;

}
