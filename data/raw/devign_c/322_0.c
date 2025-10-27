void s390_pci_sclp_configure(SCCB *sccb)

{

    PciCfgSccb *psccb = (PciCfgSccb *)sccb;

    S390PCIBusDevice *pbdev = s390_pci_find_dev_by_fid(be32_to_cpu(psccb->aid));

    uint16_t rc;



    if (be16_to_cpu(sccb->h.length) < 16) {

        rc = SCLP_RC_INSUFFICIENT_SCCB_LENGTH;

        goto out;

    }



    if (pbdev) {

        if (pbdev->configured) {

            rc = SCLP_RC_NO_ACTION_REQUIRED;

        } else {

            pbdev->configured = true;

            rc = SCLP_RC_NORMAL_COMPLETION;

        }

    } else {

        DPRINTF("sclp config no dev found\n");

        rc = SCLP_RC_ADAPTER_ID_NOT_RECOGNIZED;

    }

out:

    psccb->header.response_code = cpu_to_be16(rc);

}
