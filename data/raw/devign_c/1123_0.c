static int handle_tsch(S390CPU *cpu)

{

    CPUS390XState *env = &cpu->env;

    CPUState *cs = CPU(cpu);

    struct kvm_run *run = cs->kvm_run;

    int ret;



    cpu_synchronize_state(cs);



    ret = ioinst_handle_tsch(env, env->regs[1], run->s390_tsch.ipb);

    if (ret >= 0) {

        /* Success; set condition code. */

        setcc(cpu, ret);

        ret = 0;

    } else if (ret < -1) {

        /*

         * Failure.

         * If an I/O interrupt had been dequeued, we have to reinject it.

         */

        if (run->s390_tsch.dequeued) {

            uint16_t subchannel_id = run->s390_tsch.subchannel_id;

            uint16_t subchannel_nr = run->s390_tsch.subchannel_nr;

            uint32_t io_int_parm = run->s390_tsch.io_int_parm;

            uint32_t io_int_word = run->s390_tsch.io_int_word;

            uint32_t type = ((subchannel_id & 0xff00) << 24) |

                ((subchannel_id & 0x00060) << 22) | (subchannel_nr << 16);



            kvm_s390_interrupt_internal(cpu, type,

                                        ((uint32_t)subchannel_id << 16)

                                        | subchannel_nr,

                                        ((uint64_t)io_int_parm << 32)

                                        | io_int_word, 1);

        }

        ret = 0;

    }

    return ret;

}
