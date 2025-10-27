int svm_check_intercept_param(uint32_t type, uint64_t param)

{

    switch(type) {

    case SVM_EXIT_READ_CR0 ... SVM_EXIT_READ_CR0 + 8:

        if (INTERCEPTEDw(_cr_read, (1 << (type - SVM_EXIT_READ_CR0)))) {

            vmexit(type, param);

            return 1;

        }

        break;

    case SVM_EXIT_READ_DR0 ... SVM_EXIT_READ_DR0 + 8:

        if (INTERCEPTEDw(_dr_read, (1 << (type - SVM_EXIT_READ_DR0)))) {

            vmexit(type, param);

            return 1;

        }

        break;

    case SVM_EXIT_WRITE_CR0 ... SVM_EXIT_WRITE_CR0 + 8:

        if (INTERCEPTEDw(_cr_write, (1 << (type - SVM_EXIT_WRITE_CR0)))) {

            vmexit(type, param);

            return 1;

        }

        break;

    case SVM_EXIT_WRITE_DR0 ... SVM_EXIT_WRITE_DR0 + 8:

        if (INTERCEPTEDw(_dr_write, (1 << (type - SVM_EXIT_WRITE_DR0)))) {

            vmexit(type, param);

            return 1;

        }

        break;

    case SVM_EXIT_EXCP_BASE ... SVM_EXIT_EXCP_BASE + 16:

        if (INTERCEPTEDl(_exceptions, (1 << (type - SVM_EXIT_EXCP_BASE)))) {

            vmexit(type, param);

            return 1;

        }

        break;

    case SVM_EXIT_IOIO:

        if (INTERCEPTED(1ULL << INTERCEPT_IOIO_PROT)) {

            /* FIXME: this should be read in at vmrun (faster this way?) */

            uint64_t addr = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, control.iopm_base_pa));

            uint16_t port = (uint16_t) (param >> 16);



            if(ldub_phys(addr + port / 8) & (1 << (port % 8)))

                vmexit(type, param);

        }

        break;



    case SVM_EXIT_MSR:

        if (INTERCEPTED(1ULL << INTERCEPT_MSR_PROT)) {

            /* FIXME: this should be read in at vmrun (faster this way?) */

            uint64_t addr = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, control.msrpm_base_pa));

            switch((uint32_t)ECX) {

            case 0 ... 0x1fff:

                T0 = (ECX * 2) % 8;

                T1 = ECX / 8;

                break;

            case 0xc0000000 ... 0xc0001fff:

                T0 = (8192 + ECX - 0xc0000000) * 2;

                T1 = (T0 / 8);

                T0 %= 8;

                break;

            case 0xc0010000 ... 0xc0011fff:

                T0 = (16384 + ECX - 0xc0010000) * 2;

                T1 = (T0 / 8);

                T0 %= 8;

                break;

            default:

                vmexit(type, param);

                return 1;

            }

            if (ldub_phys(addr + T1) & ((1 << param) << T0))

                vmexit(type, param);

            return 1;

        }

        break;

    default:

        if (INTERCEPTED((1ULL << ((type - SVM_EXIT_INTR) + INTERCEPT_INTR)))) {

            vmexit(type, param);

            return 1;

        }

        break;

    }

    return 0;

}
