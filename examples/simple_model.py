def create_model(
    input_shape,
    activation="relu",
    activation_output="relu",
    kernel_initializer="glorot_normal",
    name="basic",
    size=32,
):
    model_input = Input(
        shape=input_shape,
        name=f"{name}_input",
    )
    conv_1 = Conv2D(
        size,
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_down1_conv2d",
    )(model_input)
    max_pool_1 = MaxPooling2D(padding="same")(conv_1)
    conv_2 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_down2_conv2d",
    )(max_pool_1)
    max_pool_2 = MaxPooling2D(padding="same")(conv_2)
    conv_3 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        name=f"{name}_down3_conv2d",
    )(max_pool_2)
    transpose_1 = Conv2DTranspose(
        size,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
        name=f"{name}_transpose1",
    )(conv_3)
    conv_4 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_up1_conv2d",
    )(transpose_1)
    transpose_2 = Conv2DTranspose(
        size,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
        name=f"{name}_transpose2",
    )(conv_4)
    conv_5 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_up2_conv2d",
    )(transpose_2)
    conv_6 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_up2_conv2d",
    )(conv_5)
    output = Conv2D(
        2,
        kernel_size=3,
        padding="same",
        activation=activation_output,
        kernel_initializer=kernel_initializer,
        name=f"{name}_output",
        dtype="float32",
    )(conv_5)

    return Model(
        inputs=model_input,
        outputs=clip(output, min_value=0, max_value=100),
    )