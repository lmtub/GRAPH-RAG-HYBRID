static int htab_save_iterate(QEMUFile *f, void *opaque)

{

    sPAPRMachineState *spapr = opaque;

    int fd;

    int rc = 0;



    /* Iteration header */

    if (!spapr->htab_shift) {

        qemu_put_be32(f, -1);

        return 0;

    } else {

        qemu_put_be32(f, 0);

    }



    if (!spapr->htab) {

        assert(kvm_enabled());



        fd = get_htab_fd(spapr);

        if (fd < 0) {

            return fd;

        }



        rc = kvmppc_save_htab(f, fd, MAX_KVM_BUF_SIZE, MAX_ITERATION_NS);

        if (rc < 0) {

            return rc;

        }

    } else  if (spapr->htab_first_pass) {

        htab_save_first_pass(f, spapr, MAX_ITERATION_NS);

    } else {

        rc = htab_save_later_pass(f, spapr, MAX_ITERATION_NS);

    }



    /* End marker */

    qemu_put_be32(f, 0);

    qemu_put_be16(f, 0);

    qemu_put_be16(f, 0);



    return rc;

}
