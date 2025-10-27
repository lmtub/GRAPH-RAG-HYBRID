qemu_irq *mpic_init (target_phys_addr_t base, int nb_cpus,

                        qemu_irq **irqs, qemu_irq irq_out)

{

    openpic_t *mpp;

    int i;

    struct {

        CPUReadMemoryFunc * const *read;

        CPUWriteMemoryFunc * const *write;

        target_phys_addr_t start_addr;

        ram_addr_t size;

    } const list[] = {

        {mpic_glb_read, mpic_glb_write, MPIC_GLB_REG_START, MPIC_GLB_REG_SIZE},

        {mpic_tmr_read, mpic_tmr_write, MPIC_TMR_REG_START, MPIC_TMR_REG_SIZE},

        {mpic_ext_read, mpic_ext_write, MPIC_EXT_REG_START, MPIC_EXT_REG_SIZE},

        {mpic_int_read, mpic_int_write, MPIC_INT_REG_START, MPIC_INT_REG_SIZE},

        {mpic_msg_read, mpic_msg_write, MPIC_MSG_REG_START, MPIC_MSG_REG_SIZE},

        {mpic_msi_read, mpic_msi_write, MPIC_MSI_REG_START, MPIC_MSI_REG_SIZE},

        {mpic_cpu_read, mpic_cpu_write, MPIC_CPU_REG_START, MPIC_CPU_REG_SIZE},

    };



    /* XXX: for now, only one CPU is supported */

    if (nb_cpus != 1)

        return NULL;



    mpp = g_malloc0(sizeof(openpic_t));



    for (i = 0; i < sizeof(list)/sizeof(list[0]); i++) {

        int mem_index;



        mem_index = cpu_register_io_memory(list[i].read, list[i].write, mpp,

                                           DEVICE_BIG_ENDIAN);

        if (mem_index < 0) {

            goto free;

        }

        cpu_register_physical_memory(base + list[i].start_addr,

                                     list[i].size, mem_index);

    }



    mpp->nb_cpus = nb_cpus;

    mpp->max_irq = MPIC_MAX_IRQ;

    mpp->irq_ipi0 = MPIC_IPI_IRQ;

    mpp->irq_tim0 = MPIC_TMR_IRQ;



    for (i = 0; i < nb_cpus; i++)

        mpp->dst[i].irqs = irqs[i];

    mpp->irq_out = irq_out;



    mpp->irq_raise = mpic_irq_raise;

    mpp->reset = mpic_reset;



    register_savevm(NULL, "mpic", 0, 2, openpic_save, openpic_load, mpp);

    qemu_register_reset(mpic_reset, mpp);



    return qemu_allocate_irqs(openpic_set_irq, mpp, mpp->max_irq);



free:

    g_free(mpp);

    return NULL;

}
