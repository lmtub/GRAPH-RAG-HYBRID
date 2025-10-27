static inline void gdb_continue(GDBState *s)

{

#ifdef CONFIG_USER_ONLY

    s->running_state = 1;

#else

    vm_start();

#endif

}
