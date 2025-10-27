SH7750State *sh7750_init(CPUSH4State * cpu)

{

    SH7750State *s;

    int sh7750_io_memory;

    int cpu_model = SH_CPU_SH7751R; /* for now */



    s = qemu_mallocz(sizeof(SH7750State));

    s->cpu = cpu;

    s->periph_freq = 60000000;	/* 60MHz */

    sh7750_io_memory = cpu_register_io_memory(0,

					      sh7750_mem_read,

					      sh7750_mem_write, s);

    cpu_register_physical_memory(0x1c000000, 0x04000000, sh7750_io_memory);



    sh_intc_init(&s->intc, NR_SOURCES,

		 _INTC_ARRAY(mask_registers),

		 _INTC_ARRAY(prio_registers));



    sh_intc_register_sources(&s->intc, 

			     _INTC_ARRAY(vectors),

			     _INTC_ARRAY(groups));



    sh_serial_init(0x1fe00000, 0, s->periph_freq, serial_hds[0]);

    sh_serial_init(0x1fe80000, SH_SERIAL_FEAT_SCIF,

		   s->periph_freq, serial_hds[1]);



    tmu012_init(0x1fd80000,

		TMU012_FEAT_TOCR | TMU012_FEAT_3CHAN | TMU012_FEAT_EXTCLK,

		s->periph_freq);





    if (cpu_model & (SH_CPU_SH7750 | SH_CPU_SH7750S | SH_CPU_SH7751)) {

        sh_intc_register_sources(&s->intc, 

				 _INTC_ARRAY(vectors_dma4),

				 _INTC_ARRAY(groups_dma4));

    }



    if (cpu_model & (SH_CPU_SH7750R | SH_CPU_SH7751R)) {

        sh_intc_register_sources(&s->intc, 

				 _INTC_ARRAY(vectors_dma8),

				 _INTC_ARRAY(groups_dma8));

    }



    if (cpu_model & (SH_CPU_SH7750R | SH_CPU_SH7751 | SH_CPU_SH7751R)) {

        sh_intc_register_sources(&s->intc, 

				 _INTC_ARRAY(vectors_tmu34),

				 _INTC_ARRAY(NULL));

        tmu012_init(0x1e100000, 0, s->periph_freq);

    }



    if (cpu_model & (SH_CPU_SH7751_ALL)) {

        sh_intc_register_sources(&s->intc, 

				 _INTC_ARRAY(vectors_pci),

				 _INTC_ARRAY(groups_pci));

    }



    if (cpu_model & (SH_CPU_SH7750S | SH_CPU_SH7750R | SH_CPU_SH7751_ALL)) {

        sh_intc_register_sources(&s->intc, 

				 _INTC_ARRAY(vectors_irlm),

				 _INTC_ARRAY(NULL));

    }



    return s;

}
