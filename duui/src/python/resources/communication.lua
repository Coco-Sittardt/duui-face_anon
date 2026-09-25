StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Image = luajava.bindClass("org.texttechnologylab.annotation.type.Image")

function serialize(inputCas, outputStream, parameters)
    print("start Lua serialzation")
    -- get the parameters or run default
    local hf_token = parameters["hf_token"]if parameters["hf_token"]==nil then hf_token="None" end
    local anon_type = parameters["mode"] if parameters["mode"]==nil then anon_type = "single_align" end

    -- blur/pixelate/black only work if the selected method it redaction
    -- redaction can be black, pixelate or blur
    
    local redact_type =  parameters["redact_type"] if parameters["redact_type"]==nil then redact_type = "None" end
    local blur = parameters["blur"] if parameters["blur"]==nil then blur=51 end

    local pixel = parameters["pixel"] if parameters["pixel"]==nil then pixel=16 end


    -- all the other possible settings
    local diffusion_model = parameters["diffusion_model"] if parameters["diffusion_model"]==nil then diffusion_model = "sd2-community/stable-diffusion-2-1"  end
    local clip_model = parameters["clip_model"] if parameters["clip_model"]==nil then clip_model = "openai/clip-vit-large-patch14" end
    local seed = parameters["seed"] if parameters["seed"]==nil then seed = 1 end
    local guidance = parameters["guidance"] if parameters["guidance"]==nil then guidance = 4.0 end
    local inference_steps = parameters["inference_steps"] if parameters["inference_steps"]==nil then inference_steps = 25 end
    local anon_degree = parameters["anon_degree"] if parameters["anon_degree"]==nil then anon_degree = 1.25 end
    -- swap has different default settings
    if anon_type == "swap" then
        inference_steps = 200
        anon_degree = 0.0

    end
    local vis_input = parameters["vis_input"] if parameters["vis_input"]==nil then vis_input = "False" end
    local height = parameters["height"] 
    --if parameters["height"]==nil then height = tonumber(height) end
    local width = parameters["width"] 
    --if parameters["width"]==nil then width = tonumber(width) end


    print(anon_type..redact_type..blur..pixel..diffusion_model..clip_model..seed..guidance..inference_steps..anon_degree..vis_input..tostring(height)..tostring(width))
     


    -- input images

    local images = {}
    local found = JCasUtil:select(inputCas, Image)

    for i=0, found:size()-1 do
        local image = found:get(i)
        images[tostring(i + 1)] = {
            src = image:getSrc(),
            height = image:getHeight(),
            width = image:getWidth(),
            begin = image:getBegin(),
            ['end'] = image:getEnd()
        }
    end
    local number_of_images = found:size()
    print(number_of_images)
    outputStream:write(json.encode({
        anon_type = anon_type,
        anon_degree = anon_degree,
        images = images,
        redact_type = redact_type,
        blur = blur,
        pixel = pixel,
        diffusion_model = diffusion_model,
        clip_model = clip_model,
        seed = seed,
        guidance = guidance,
        inference_steps =inference_steps,
        vis_input = vis_input,
        height =height,
        width = width,
        hf_token=hf_token
    }))
    
end

function deserialize(inputCas, inputStream)
    print("start deserialize")
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
    print(results)

    if results['out_errors'] ~= nil then
        local errors = results['out_errors']
        for index_i, error in ipairs(errors) do
            local warning_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            warning_i:setKey("error")
            warning_i:setValue(error)
            warning_i:addToIndexes()
        end
    end

    -- anonymiyed images
    if results['output_images'] ~= nil then
        local output_images = results['output_images']
        for image_id, img_data in pairs(output_images) do
            local image = luajava.newInstance("org.texttechnologylab.annotation.type.Image", inputCas)
            image:setSrc(img_data["src"])
            image:setWidth(img_data['width'])
            image:setHeight(img_data['height'])
            image:setBegin(img_data['begin'])
            image:setEnd(img_data['end'])
            image:addToIndexes()
        end
    end

end
    --print("---------------------- Finished errors ----------------------")
    
