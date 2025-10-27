void s390_io_interrupt(S390CPU *cpu, uint16_t subchannel_id,

                       uint16_t subchannel_nr, uint32_t io_int_parm,

                       uint32_t io_int_word)

{

    if (kvm_enabled()) {

        kvm_s390_io_interrupt(cpu, subchannel_id, subchannel_nr, io_int_parm,

                              io_int_word);

    } else {

        cpu_inject_io(cpu, subchannel_id, subchannel_nr, io_int_parm,

                      io_int_word);

    }

}
