static uint32_t s390_pci_generate_fid(Error **errp)

{

    uint32_t fid = 0;



    while (fid <= ZPCI_MAX_FID) {

        if (!s390_pci_find_dev_by_fid(fid)) {

            return fid;

        }



        if (fid == ZPCI_MAX_FID) {

            break;

        }



        fid++;

    }



    error_setg(errp, "no free fid could be found");

    return 0;

}
