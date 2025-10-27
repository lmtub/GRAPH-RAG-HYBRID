static target_ulong h_random(PowerPCCPU *cpu, sPAPRMachineState *spapr,

                             target_ulong opcode, target_ulong *args)

{

    sPAPRRngState *rngstate;

    HRandomData hrdata;



    rngstate = SPAPR_RNG(object_resolve_path_type("", TYPE_SPAPR_RNG, NULL));



    if (!rngstate || !rngstate->backend) {

        return H_HARDWARE;

    }



    qemu_sem_init(&hrdata.sem, 0);

    hrdata.val.v64 = 0;

    hrdata.received = 0;



    qemu_mutex_unlock_iothread();

    while (hrdata.received < 8) {

        rng_backend_request_entropy(rngstate->backend, 8 - hrdata.received,

                                    random_recv, &hrdata);

        qemu_sem_wait(&hrdata.sem);

    }

    qemu_mutex_lock_iothread();



    qemu_sem_destroy(&hrdata.sem);

    args[0] = hrdata.val.v64;



    return H_SUCCESS;

}
