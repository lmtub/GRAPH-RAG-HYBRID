static int dxva2_device_create9ex(AVHWDeviceContext *ctx, UINT adapter)

{

    DXVA2DevicePriv *priv = ctx->user_opaque;

    D3DPRESENT_PARAMETERS d3dpp = dxva2_present_params;

    D3DDISPLAYMODEEX modeex = {0};

    IDirect3D9Ex *d3d9ex = NULL;

    IDirect3DDevice9Ex *exdev = NULL;

    HRESULT hr;

    pDirect3DCreate9Ex *createD3DEx = (pDirect3DCreate9Ex *)dlsym(priv->d3dlib, "Direct3DCreate9Ex");

    if (!createD3DEx)

        return AVERROR(ENOSYS);



    hr = createD3DEx(D3D_SDK_VERSION, &d3d9ex);

    if (FAILED(hr))

        return AVERROR_UNKNOWN;



    IDirect3D9Ex_GetAdapterDisplayModeEx(d3d9ex, adapter, &modeex, NULL);



    d3dpp.BackBufferFormat = modeex.Format;



    hr = IDirect3D9Ex_CreateDeviceEx(d3d9ex, adapter, D3DDEVTYPE_HAL, GetDesktopWindow(),

                                     FF_D3DCREATE_FLAGS,

                                     &d3dpp, NULL, &exdev);

    if (FAILED(hr)) {

        IDirect3D9Ex_Release(d3d9ex);

        return AVERROR_UNKNOWN;

    }



    av_log(ctx, AV_LOG_VERBOSE, "Using D3D9Ex device.\n");

    priv->d3d9 = (IDirect3D9 *)d3d9ex;

    priv->d3d9device = (IDirect3DDevice9 *)exdev;

    return 0;

}
