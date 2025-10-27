int clp_service_call(S390CPU *cpu, uint8_t r2)

{

    ClpReqHdr *reqh;

    ClpRspHdr *resh;

    S390PCIBusDevice *pbdev;

    uint32_t req_len;

    uint32_t res_len;

    uint8_t buffer[4096 * 2];

    uint8_t cc = 0;

    CPUS390XState *env = &cpu->env;

    int i;



    cpu_synchronize_state(CPU(cpu));



    if (env->psw.mask & PSW_MASK_PSTATE) {

        program_interrupt(env, PGM_PRIVILEGED, 4);

        return 0;

    }



    if (s390_cpu_virt_mem_read(cpu, env->regs[r2], r2, buffer, sizeof(*reqh))) {

        return 0;

    }

    reqh = (ClpReqHdr *)buffer;

    req_len = lduw_p(&reqh->len);

    if (req_len < 16 || req_len > 8184 || (req_len % 8 != 0)) {

        program_interrupt(env, PGM_OPERAND, 4);

        return 0;

    }



    if (s390_cpu_virt_mem_read(cpu, env->regs[r2], r2, buffer,

                               req_len + sizeof(*resh))) {

        return 0;

    }

    resh = (ClpRspHdr *)(buffer + req_len);

    res_len = lduw_p(&resh->len);

    if (res_len < 8 || res_len > 8176 || (res_len % 8 != 0)) {

        program_interrupt(env, PGM_OPERAND, 4);

        return 0;

    }

    if ((req_len + res_len) > 8192) {

        program_interrupt(env, PGM_OPERAND, 4);

        return 0;

    }



    if (s390_cpu_virt_mem_read(cpu, env->regs[r2], r2, buffer,

                               req_len + res_len)) {

        return 0;

    }



    if (req_len != 32) {

        stw_p(&resh->rsp, CLP_RC_LEN);

        goto out;

    }



    switch (lduw_p(&reqh->cmd)) {

    case CLP_LIST_PCI: {

        ClpReqRspListPci *rrb = (ClpReqRspListPci *)buffer;

        list_pci(rrb, &cc);

        break;

    }

    case CLP_SET_PCI_FN: {

        ClpReqSetPci *reqsetpci = (ClpReqSetPci *)reqh;

        ClpRspSetPci *ressetpci = (ClpRspSetPci *)resh;



        pbdev = s390_pci_find_dev_by_fh(ldl_p(&reqsetpci->fh));

        if (!pbdev) {

                stw_p(&ressetpci->hdr.rsp, CLP_RC_SETPCIFN_FH);

                goto out;

        }



        switch (reqsetpci->oc) {

        case CLP_SET_ENABLE_PCI_FN:

            pbdev->fh |= FH_MASK_ENABLE;

            pbdev->state = ZPCI_FS_ENABLED;

            stl_p(&ressetpci->fh, pbdev->fh);

            stw_p(&ressetpci->hdr.rsp, CLP_RC_OK);

            break;

        case CLP_SET_DISABLE_PCI_FN:

            pbdev->fh &= ~FH_MASK_ENABLE;

            pbdev->state = ZPCI_FS_DISABLED;

            stl_p(&ressetpci->fh, pbdev->fh);

            stw_p(&ressetpci->hdr.rsp, CLP_RC_OK);

            break;

        default:

            DPRINTF("unknown set pci command\n");

            stw_p(&ressetpci->hdr.rsp, CLP_RC_SETPCIFN_FHOP);

            break;

        }

        break;

    }

    case CLP_QUERY_PCI_FN: {

        ClpReqQueryPci *reqquery = (ClpReqQueryPci *)reqh;

        ClpRspQueryPci *resquery = (ClpRspQueryPci *)resh;



        pbdev = s390_pci_find_dev_by_fh(ldl_p(&reqquery->fh));

        if (!pbdev) {

            DPRINTF("query pci no pci dev\n");

            stw_p(&resquery->hdr.rsp, CLP_RC_SETPCIFN_FH);

            goto out;

        }



        for (i = 0; i < PCI_BAR_COUNT; i++) {

            uint32_t data = pci_get_long(pbdev->pdev->config +

                PCI_BASE_ADDRESS_0 + (i * 4));



            stl_p(&resquery->bar[i], data);

            resquery->bar_size[i] = pbdev->pdev->io_regions[i].size ?

                                    ctz64(pbdev->pdev->io_regions[i].size) : 0;

            DPRINTF("bar %d addr 0x%x size 0x%" PRIx64 "barsize 0x%x\n", i,

                    ldl_p(&resquery->bar[i]),

                    pbdev->pdev->io_regions[i].size,

                    resquery->bar_size[i]);

        }



        stq_p(&resquery->sdma, ZPCI_SDMA_ADDR);

        stq_p(&resquery->edma, ZPCI_EDMA_ADDR);

        stl_p(&resquery->fid, pbdev->fid);

        stw_p(&resquery->pchid, 0);

        stw_p(&resquery->ug, 1);

        stl_p(&resquery->uid, pbdev->fid);

        stw_p(&resquery->hdr.rsp, CLP_RC_OK);

        break;

    }

    case CLP_QUERY_PCI_FNGRP: {

        ClpRspQueryPciGrp *resgrp = (ClpRspQueryPciGrp *)resh;

        resgrp->fr = 1;

        stq_p(&resgrp->dasm, 0);

        stq_p(&resgrp->msia, ZPCI_MSI_ADDR);

        stw_p(&resgrp->mui, 0);

        stw_p(&resgrp->i, 128);

        resgrp->version = 0;



        stw_p(&resgrp->hdr.rsp, CLP_RC_OK);

        break;

    }

    default:

        DPRINTF("unknown clp command\n");

        stw_p(&resh->rsp, CLP_RC_CMD);

        break;

    }



out:

    if (s390_cpu_virt_mem_write(cpu, env->regs[r2], r2, buffer,

                                req_len + res_len)) {

        return 0;

    }

    setcc(cpu, cc);

    return 0;

}
