int fread_targphys_ok(target_phys_addr_t dst_addr, size_t nbytes, FILE *f)

{

    return fread_targphys(dst_addr, nbytes, f) == nbytes;

}
