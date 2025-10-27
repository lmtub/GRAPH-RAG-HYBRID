static void nvdimm_build_device_dsm(Aml *dev)

{

    Aml *method;



    method = aml_method("_DSM", 4, AML_NOTSERIALIZED);

    aml_append(method, aml_return(aml_call4(NVDIMM_COMMON_DSM, aml_arg(0),

                                  aml_arg(1), aml_arg(2), aml_arg(3))));

    aml_append(dev, method);

}
