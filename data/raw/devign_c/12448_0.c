static inline void clear_float_exceptions(CPUSPARCState *env)

{

    set_float_exception_flags(0, &env->fp_status);

}
