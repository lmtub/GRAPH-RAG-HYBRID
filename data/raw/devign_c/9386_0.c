void css_adapter_interrupt(uint8_t isc)

{

    S390CPU *cpu = s390_cpu_addr2state(0);

    uint32_t io_int_word = (isc << 27) | IO_INT_WORD_AI;



    trace_css_adapter_interrupt(isc);

    s390_io_interrupt(cpu, 0, 0, 0, io_int_word);

}
