struct icp_state *xics_system_init(int nr_irqs)

{

    CPUPPCState *env;

    CPUState *cpu;

    int max_server_num;

    struct icp_state *icp;

    struct ics_state *ics;



    max_server_num = -1;

    for (env = first_cpu; env != NULL; env = env->next_cpu) {

        cpu = CPU(ppc_env_get_cpu(env));

        if (cpu->cpu_index > max_server_num) {

            max_server_num = cpu->cpu_index;

        }

    }



    icp = g_malloc0(sizeof(*icp));

    icp->nr_servers = max_server_num + 1;

    icp->ss = g_malloc0(icp->nr_servers*sizeof(struct icp_server_state));



    for (env = first_cpu; env != NULL; env = env->next_cpu) {

        cpu = CPU(ppc_env_get_cpu(env));

        struct icp_server_state *ss = &icp->ss[cpu->cpu_index];



        switch (PPC_INPUT(env)) {

        case PPC_FLAGS_INPUT_POWER7:

            ss->output = env->irq_inputs[POWER7_INPUT_INT];

            break;



        case PPC_FLAGS_INPUT_970:

            ss->output = env->irq_inputs[PPC970_INPUT_INT];

            break;



        default:

            hw_error("XICS interrupt model does not support this CPU bus "

                     "model\n");

            exit(1);

        }

    }



    ics = g_malloc0(sizeof(*ics));

    ics->nr_irqs = nr_irqs;

    ics->offset = XICS_IRQ_BASE;

    ics->irqs = g_malloc0(nr_irqs * sizeof(struct ics_irq_state));

    ics->islsi = g_malloc0(nr_irqs * sizeof(bool));



    icp->ics = ics;

    ics->icp = icp;



    ics->qirqs = qemu_allocate_irqs(ics_set_irq, ics, nr_irqs);



    spapr_register_hypercall(H_CPPR, h_cppr);

    spapr_register_hypercall(H_IPI, h_ipi);

    spapr_register_hypercall(H_XIRR, h_xirr);

    spapr_register_hypercall(H_EOI, h_eoi);



    spapr_rtas_register("ibm,set-xive", rtas_set_xive);

    spapr_rtas_register("ibm,get-xive", rtas_get_xive);

    spapr_rtas_register("ibm,int-off", rtas_int_off);

    spapr_rtas_register("ibm,int-on", rtas_int_on);



    qemu_register_reset(xics_reset, icp);



    return icp;

}
