void g_free(void *ptr)

{

    __coverity_free__(ptr);

    __coverity_mark_as_afm_freed__(ptr, AFM_free);

}
