void vring_teardown(Vring *vring)

{

    hostmem_finalize(&vring->hostmem);

}
