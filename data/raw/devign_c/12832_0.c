void op_flush_icache_all(void) {

    CALL_FROM_TB1(tb_flush, env);

    RETURN();

}
