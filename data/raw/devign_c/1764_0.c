uint32_t kvmppc_get_vmx(void)

{

    return kvmppc_read_int_cpu_dt("ibm,vmx");

}
