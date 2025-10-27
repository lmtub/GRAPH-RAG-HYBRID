CPUDebugExcpHandler *cpu_set_debug_excp_handler(CPUDebugExcpHandler *handler)

{

    CPUDebugExcpHandler *old_handler = debug_excp_handler;



    debug_excp_handler = handler;

    return old_handler;

}
