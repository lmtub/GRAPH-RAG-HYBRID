void kvm_s390_io_interrupt(S390CPU *cpu, uint16_t subchannel_id,

                           uint16_t subchannel_nr, uint32_t io_int_parm,

                           uint32_t io_int_word)

{

    uint32_t type;



    if (io_int_word & IO_INT_WORD_AI) {

        type = KVM_S390_INT_IO(1, 0, 0, 0);

    } else {

        type = ((subchannel_id & 0xff00) << 24) |

            ((subchannel_id & 0x00060) << 22) | (subchannel_nr << 16);

    }

    kvm_s390_interrupt_internal(cpu, type,

                                ((uint32_t)subchannel_id << 16) | subchannel_nr,

                                ((uint64_t)io_int_parm << 32) | io_int_word, 1);

}
