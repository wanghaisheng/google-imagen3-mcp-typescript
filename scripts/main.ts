import { Buffer } from 'buffer';
import { format as formatDate } from 'date-fns';
import envPaths from 'env-paths';
import { nanoid } from 'nanoid';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import winston from 'winston';
import 'winston-daily-rotate-file';
import express from 'express';
import cors from 'cors';
import * as readline from 'readline';

//region Type Definitions (from Rust structs)

interface ImagePrompt {
    /**
     * The prompt text for image generation. The prompt MUST be in English.
     */
    prompt: string;

    /**
     * The aspect ratio of the image to generate. Supported values are "1:1", "3:4", "4:3", "9:16", and "16:9". The default is "1:1".
     */
    aspect_ratio?: string;
}

// Request and response structures for the Gemini API
interface GeminiRequest {
    instances: GeminiInstance[];
    parameters: GeminiParameters;
}

interface GeminiInstance {
    prompt: string;
}

interface GeminiParameters {
    sampleCount: number;
    aspectRatio?: string;
}

interface GeminiResponse {
    predictions?: GeminiPrediction[];
    error?: { message: string };
}

interface GeminiPrediction {
    mimeType: string;
    bytesBase64Encoded: string;
}

// Equivalent of Rust's ServerInfo for RMCP
interface ServerInfo {
    server_info: {
        name: string;
        version: string;
    };
    instructions?: string;
    capabilities: {
        tools: boolean;
    };
}

//endregion

class ImageGenerationServer {
    private readonly resourcesPath: string;
    private readonly imageResourceServerAddr: string;
    private readonly serverPort: number;

    constructor(resourcesPath: string, imageResourceServerAddr: string, serverPort: number) {
        this.resourcesPath = resourcesPath;
        this.imageResourceServerAddr = imageResourceServerAddr;
        this.serverPort = serverPort;
    }

    /**
     * Generate an image based on a prompt. Returns an image URL that can be used in markdown format like ![description](URL) to display the image
     */
    async generate_image(args: ImagePrompt): Promise<string> {
        winston.info('Received image generation request', { args });

        const SUPPORTED_ASPECT_RATIOS = ["1:1", "3:4", "4:3", "9:16", "16:9"];
        if (args.aspect_ratio && !SUPPORTED_ASPECT_RATIOS.includes(args.aspect_ratio)) {
            const errorMsg = `Invalid aspect ratio: ${args.aspect_ratio}, supported values are: ${SUPPORTED_ASPECT_RATIOS.join(", ")}`;
            winston.error(errorMsg);
            return errorMsg;
        }

        try {
            const filenames = await generateImageFromGemini(
                args.prompt,
                args.aspect_ratio,
                this.resourcesPath
            );

            const urls = filenames.map(filename =>
                `http://${this.imageResourceServerAddr}:${this.serverPort}/images/${filename}`
            ).join("\n");
            
            winston.info(`Image generation successful. ${filenames.length} image(s) created.`);
            return urls;

        } catch (e: any) {
            winston.error(`Error generating image: ${e.message}`, { error: e });
            return `Error generating image: ${e.message}`;
        }
    }

    /**
     * Provides information about the server and its capabilities, including detailed instructions for the tool.
     */
    getInfo(): ServerInfo {
        return {
            server_info: {
                name: "imagen3-mcp",
                version: "0.1.0",
            },
            instructions: `
Use the generate_image tool to create images from text descriptions. The returned URL can be used in markdown format like ![description](URL) to display the image.

Before generating an image, please read the <Imagen_prompt_guide> section to understand how to create effective prompts.

<Imagen_prompt_guide>
## Prompt writing basics
Description of the image to generate. Maximum prompt length is 480 tokens. A good prompt is descriptive and clear, and makes use of meaningful keywords and modifiers. Start by thinking of your subject, context, and style.
Example Prompt: A sketch (style) of a modern apartment building (subject) surrounded by skyscrapers (context and background).
1. Subject: The first thing to think about with any prompt is the subject: the object, person, animal, or scenery you want an image of.
2. Context and background: Just as important is the background or context in which the subject will be placed. Try placing your subject in a variety of backgrounds. For example, a studio with a white background, outdoors, or indoor environments.
3. Style: Finally, add the style of image you want. Styles can be general (painting, photograph, sketches) or very specific (pastel painting, charcoal drawing, isometric 3D). You can also combine styles.
After you write a first version of your prompt, refine your prompt by adding more details until you get to the image that you want. Iteration is important. Start by establishing your core idea, and then refine and expand upon that core idea until the generated image is close to your vision.
Imagen 3 can transform your ideas into detailed images, whether your prompts are short or long and detailed. Refine your vision through iterative prompting, adding details until you achieve the perfect result.
Example Prompt: close-up photo of a woman in her 20s, street photography, movie still, muted orange warm tones
Example Prompt: captivating photo of a woman in her 20s utilizing a street photography style. The image should look like a movie still with muted orange warm tones.
Additional advice for Imagen prompt writing:
- Use descriptive language: Employ detailed adjectives and adverbs to paint a clear picture for Imagen 3.
- Provide context: If necessary, include background information to aid the AI's understanding.
- Reference specific artists or styles: If you have a particular aesthetic in mind, referencing specific artists or art movements can be helpful.
- Use prompt engineering tools: Consider exploring prompt engineering tools or resources to help you refine your prompts and achieve optimal results.
- Enhancing the facial details in your personal and group images: Specify facial details as a focus of the photo (for example, use the word "portrait" in the prompt).
## Generate text in images
Imagen can add text into images, opening up more creative image generation possibilities. Use the following guidance to get the most out of this feature:
- Iterate with confidence: You might have to regenerate images until you achieve the look you want. Imagen's text integration is still evolving, and sometimes multiple attempts yield the best results.
- Keep it short: Limit text to 25 characters or less for optimal generation.
- Multiple phrases: Experiment with two or three distinct phrases to provide additional information. Avoid exceeding three phrases for cleaner compositions.
Example Prompt: A poster with the text "Summerland" in bold font as a title, underneath this text is the slogan "Summer never felt so good"
- Guide Placement: While Imagen can attempt to position text as directed, expect occasional variations. This feature is continually improving.
- Inspire font style: Specify a general font style to subtly influence Imagen's choices. Don't rely on precise font replication, but expect creative interpretations.
- Font size: Specify a font size or a general indication of size (for example, small, medium, large) to influence the font size generation.
## Advanced prompt writing techniques
Use the following examples to create more specific prompts based on attributes like photography descriptors, shapes and materials, historical art movements, and image quality modifiers.
### Photography
- Prompt includes: "A photo of..."
To use this style, start with using keywords that clearly tell Imagen that you're looking for a photograph. Start your prompts with "A photo of. . .". For example:
Example Prompt: A photo of coffee beans in a kitchen on a wooden surface
Example Prompt: A photo of a chocolate bar on a kitchen counter
Example Prompt: A photo of a modern building with water in the background
#### Photography modifiers
In the following examples, you can see several photography-specific modifiers and parameters. You can combine multiple modifiers for more precise control.
1. Camera Proximity - Close up, taken from far away
   Example Prompt: A close-up photo of coffee beans
   Example Prompt: A zoomed out photo of a small bag of coffee beans in a messy kitchen
2. Camera Position - aerial, from below
   Example Prompt: aerial photo of urban city with skyscrapers
   Example Prompt: A photo of a forest canopy with blue skies from below
3. Lighting - natural, dramatic, warm, cold
   Example Prompt: studio photo of a modern arm chair, natural lighting
   Example Prompt: studio photo of a modern arm chair, dramatic lighting
4. Camera Settings - motion blur, soft focus, bokeh, portrait
   Example Prompt: photo of a city with skyscrapers from the inside of a car with motion blur
   Example Prompt: soft focus photograph of a bridge in an urban city at night
5. Lens types - 35mm, 50mm, fisheye, wide angle, macro
   Example Prompt: photo of a leaf, macro lens
   Example Prompt: street photography, new york city, fisheye lens
6. Film types - black and white, polaroid
   Example Prompt: a polaroid portrait of a dog wearing sunglasses
   Example Prompt: black and white photo of a dog wearing sunglasses
### Illustration and art
- Prompt includes: "A painting of...", "A sketch of..."
Art styles vary from monochrome styles like pencil sketches, to hyper-realistic digital art. For example, the following images use the same prompt with different styles:
"An [art style or creation technique] of an angular sporty electric sedan with skyscrapers in the background"
Example Prompt: A technical pencil drawing of an angular...
Example Prompt: A charcoal drawing of an angular...
Example Prompt: A color pencil drawing of an angular...
Example Prompt: A pastel painting of an angular...
Example Prompt: A digital art of an angular...
Example Prompt: An art deco (poster) of an angular...
#### Shapes and materials
- Prompt includes: "...made of...", "...in the shape of..."
One of the strengths of this technology is that you can create imagery that is otherwise difficult or impossible. For example, you can recreate your company logo in different materials and textures.
Example Prompt: a duffle bag made of cheese
Example Prompt: neon tubes in the shape of a bird
Example Prompt: an armchair made of paper, studio photo, origami style
#### Historical art references
- Prompt includes: "...in the style of..."
Certain styles have become iconic over the years. The following are some ideas of historical painting or art styles that you can try.
"generate an image in the style of [art period or movement] : a wind farm"
Example Prompt: generate an image in the style of an impressionist painting: a wind farm
Example Prompt: generate an image in the style of a renaissance painting: a wind farm
Example Prompt: generate an image in the style of pop art: a wind farm
### Image quality modifiers
Certain keywords can let the model know that you're looking for a high-quality asset. Examples of quality modifiers include the following:
- General Modifiers - high-quality, beautiful, stylized
- Photos - 4K, HDR, Studio Photo
- Art, Illustration - by a professional, detailed
The following are a few examples of prompts without quality modifiers and the same prompt with quality modifiers.
Example Prompt: (no quality modifiers): a photo of a corn stalk
Example Prompt: (with quality modifiers): 4k HDR beautiful photo of a corn stalk taken by a professional photographer
### Aspect ratios
Imagen 3 image generation lets you set five distinct image aspect ratios.
1. Square (1:1, default) - A standard square photo. Common uses for this aspect ratio include social media posts.
2. Fullscreen (4:3) - This aspect ratio is commonly used in media or film. It is also the dimensions of most old (non-widescreen) TVs and medium format cameras. It captures more of the scene horizontally (compared to 1:1), making it a preferred aspect ratio for photography.
   Example Prompt: close up of a musician's fingers playing the piano, black and white film, vintage (4:3 aspect ratio)
   Example Prompt: A professional studio photo of french fries for a high end restaurant, in the style of a food magazine (4:3 aspect ratio)
3. Portrait full screen (3:4) - This is the fullscreen aspect ratio rotated 90 degrees. This lets to capture more of the scene vertically compared to the 1:1 aspect ratio.
   Example Prompt: a woman hiking, close of her boots reflected in a puddle, large mountains in the background, in the style of an advertisement, dramatic angles (3:4 aspect ratio)
   Example Prompt: aerial shot of a river flowing up a mystical valley (3:4 aspect ratio)
4. Widescreen (16:9) - This ratio has replaced 4:3 and is now the most common aspect ratio for TVs, monitors, and mobile phone screens (landscape). Use this aspect ratio when you want to capture more of the background (for example, scenic landscapes).
   Example Prompt: a man wearing all white clothing sitting on the beach, close up, golden hour lighting (16:9 aspect ratio)
5. Portrait (9:16) - This ratio is widescreen but rotated. This a relatively new aspect ratio that has been popularized by short form video apps (for example, YouTube shorts). Use this for tall objects with strong vertical orientations such as buildings, trees, waterfalls, or other similar objects.
   Example Prompt: a digital render of a massive skyscraper, modern, grand, epic with a beautiful sunset in the background (9:16 aspect ratio)
### Photorealistic images
Different versions of the image generation model might offer a mix of artistic and photorealistic output. Use the following wording in prompts to generate more photorealistic output, based on the subject you want to generate.
Note: Take these keywords as general guidance when you try to create photorealistic images. They aren't required to achieve your goal.
| Use case | Lens type | Focal lengths | Additional details |
| --- | --- | --- | --- |
| People (portraits) | Prime, zoom | 24-35mm | black and white film, Film noir, Depth of field, duotone (mention two colors) |
| Food, insects, plants (objects, still life) | Macro | 60-105mm | High detail, precise focusing, controlled lighting |
| Sports, wildlife (motion) | Telephoto zoom | 100-400mm | Fast shutter speed, Action or movement tracking |
| Astronomical, landscape (wide-angle) | Wide-angle | 10-24mm | Long exposure times, sharp focus, long exposure, smooth water or clouds |
#### Portraits
| Use case | Lens type | Focal lengths | Additional details |
| --- | --- | --- | --- |
| People (portraits) | Prime, zoom | 24-35mm | black and white film, Film noir, Depth of field, duotone (mention two colors) |
Using several keywords from the table, Imagen can generate the following portraits:
Example Prompt: A woman, 35mm portrait, blue and grey duotones
Example Prompt: A woman, 35mm portrait, film noir
#### Objects:
| Use case | Lens type | Focal lengths | Additional details |
| --- | --- | --- | --- |
| Food, insects, plants (objects, still life) | Macro | 60-105mm | High detail, precise focusing, controlled lighting |
Using several keywords from the table, Imagen can generate the following object images:
Example Prompt: leaf of a prayer plant, macro lens, 60mm
Example Prompt: a plate of pasta, 100mm Macro lens
#### Motion
| Use case | Lens type | Focal lengths | Additional details |
| --- | --- | --- | --- |
| Sports, wildlife (motion) | Telephoto zoom | 100-400mm | Fast shutter speed, Action or movement tracking |
Using several keywords from the table, Imagen can generate the following motion images:
Example Prompt: a winning touchdown, fast shutter speed, movement tracking
Example Prompt: A deer running in the forest, fast shutter speed, movement tracking
#### Wide-angle
| Use case | Lens type | Focal lengths | Additional details |
| --- | --- | --- | --- |
| Astronomical, landscape (wide-angle) | Wide-angle | 10-24mm | Long exposure times, sharp focus, long exposure, smooth water or clouds |
Using several keywords from the table, Imagen can generate the following wide-angle images:
Example Prompt: an expansive mountain range, landscape wide angle 10mm
Example Prompt: a photo of the moon, astro photography, wide angle 10mm
</Imagen_prompt_guide>
`.trim(),
            capabilities: {
                tools: true,
            },
        };
    }
}

//region Utility Functions

/**
 * Generates an image using the Gemini API.
 */
async function generateImageFromGemini(
    prompt: string,
    aspectRatio: string | undefined,
    resourcesPath: string
): Promise<string[]> {
    winston.info('Generating image from Gemini', { prompt, aspectRatio, prompt_length: prompt.length });

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
        throw new Error("GEMINI_API_KEY environment variable not set");
    }

    const request: GeminiRequest = {
        instances: [{ prompt }],
        parameters: {
            sampleCount: 1, // Just generate one image
            aspectRatio: aspectRatio,
        },
    };
    winston.info('Sending request to Gemini', { request });

    const baseUrl = process.env.BASE_URL || "https://generativelanguage.googleapis.com";
    const url = `${baseUrl}/v1beta/models/imagen-3.0-generate-002:predict?key=${apiKey}`;

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
    });

    const responseText = await response.text();
    if (!response.ok) {
        winston.error('Failed to send request to Gemini', { status: response.status, body: responseText });
        throw new Error(`Gemini API request failed with status ${response.status}: ${responseText}`);
    }

    let geminiResponse: GeminiResponse;
    try {
        geminiResponse = JSON.parse(responseText);
    } catch (e: any) {
        winston.error('Failed to parse Gemini response', { response_body: responseText, error: e });
        throw new Error(`Failed to parse Gemini response: ${e.message}. The response was: ${responseText}`);
    }
    
    if (geminiResponse.error) {
        winston.error('Gemini API returned an error', { error: geminiResponse.error });
        throw new Error(`Gemini API Error: ${geminiResponse.error.message}`);
    }

    const predictions = geminiResponse.predictions ?? [];
    if (predictions.length === 0) {
        winston.error("No images were generated by Gemini. This might be due to safety filters.");
        throw new Error("No images were generated. This might be due to the image not passing Google's safety review.");
    }

    const filenames: string[] = [];
    const imagesDir = path.join(resourcesPath, "images");

    for (const pred of predictions) {
        const timestamp = formatDate(new Date(), 'yyyyMMddHHmmss');
        const id = nanoid(10);
        const filename = `${id}_${timestamp}.png`;
        const filePath = path.join(imagesDir, filename);

        try {
            const imageData = Buffer.from(pred.bytesBase64Encoded, 'base64');
            await fs.writeFile(filePath, imageData);
            winston.info(`Successfully saved generated image.`, { file_path: filePath });
            filenames.push(filename);
        } catch (e: any) {
            winston.error(`Failed to decode base64 or write image to disk: ${e.message}`, { file_path: filePath });
            throw e; // Propagate the error
        }
    }
    return filenames;
}

/**
 * Ensures a directory exists, creating it if necessary.
 * Corresponds to Rust's ProjectDirs and fs::create_dir_all
 */
async function ensureDir(dirPath: string, purpose: string): Promise<void> {
    try {
        await fs.mkdir(dirPath, { recursive: true });
    } catch (e: any) {
        if (e.code !== 'EEXIST') { // Ignore error if directory already exists
            throw new Error(`Could not create ${purpose} directory at ${dirPath}: ${e.message}`);
        }
    }
}

/**
 * Handler to list images in the images directory.
 */
async function listImages(resourcesPath: string): Promise<string[]> {
    const imagesDir = path.join(resourcesPath, "images");
    const entries = await fs.readdir(imagesDir, { withFileTypes: true });
    return entries
        .filter(dirent => dirent.isFile())
        .map(dirent => dirent.name);
}
//endregion

// --- Main Application Entry Point ---
async function main() {
    // --- App Path Setup ---
    const paths = envPaths("imagen3-mcp", { suffix: "" }); // Using 'imagen3-mcp' as the app name
    const dataDir = paths.data; // Cross-platform data directory
    const logDir = path.join(dataDir, 'logs');
    const resourcesPath = path.join(dataDir, 'artifacts');
    const imagesDir = path.join(resourcesPath, 'images');

    // --- Logging Setup (like Tracing) ---
    await ensureDir(logDir, 'log');
    const logFilePrefix = "imagen3-mcp.log";
    
    winston.configure({
        level: process.env.LOG_LEVEL || 'info',
        format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json()
        ),
        transports: [
            new winston.transports.Console({
                format: winston.format.combine(
                    winston.format.colorize(),
                    winston.format.simple()
                )
            }),
            new winston.transports.DailyRotateFile({
                filename: path.join(logDir, `${logFilePrefix}-%DATE%.log`),
                datePattern: 'YYYY-MM-DD',
                zippedArchive: true,
                maxSize: '20m',
                maxFiles: '14d',
            }),
        ],
    });
    winston.info(`Tracing initialized. Logging to console and ${logDir}`);

    // --- Directory Setup ---
    try {
        await ensureDir(resourcesPath, 'resources');
        winston.info('Created resources directory.', { path: resourcesPath });
        await ensureDir(imagesDir, 'images');
        winston.info('Created images directory.', { path: imagesDir });
    } catch (e: any) {
        winston.error(`Failed to ensure resources directory: ${e.message}`);
        process.exit(1);
    }
    
    // --- Configuration ---
    const imageResourceServerAddr = process.env.IMAGE_RESOURCE_SERVER_ADDR || '127.0.0.1';
    const serverPort = parseInt(process.env.SERVER_PORT || '9981', 10);
    const listenAddr = process.env.SERVER_LISTEN_ADDR || '127.0.0.1';

    if (isNaN(serverPort)) {
        winston.error(`Invalid SERVER_PORT: ${process.env.SERVER_PORT}`);
        process.exit(1);
    }
    
    if (!process.env.GEMINI_API_KEY) {
        winston.error("GEMINI_API_KEY environment variable is not set. Image generation will fail.");
        process.exit(1);
    } else {
        winston.info("GEMINI_API_KEY found.");
    }

    // --- HTTP Server (like Warp) ---
    const app = express();
    app.use(cors());

    winston.info(`Serving images from directory`, { path: imagesDir });
    app.use('/images', express.static(imagesDir));

    app.get('/list-images', async (req, res) => {
        winston.info("Received request to list images.");
        try {
            const imageList = await listImages(resourcesPath);
            res.json(imageList);
        } catch (e: any) {
            winston.error(`Failed to list images: ${e.message}`);
            res.status(500).send("Failed to list images.");
        }
    });

    const httpServer = app.listen(serverPort, listenAddr, () => {
        winston.info(`Starting HTTP server for image resources.`, { address: `http://${listenAddr}:${serverPort}`});
    });

    // --- MCP Server (stdin/stdout) ---
    const service = new ImageGenerationServer(
        resourcesPath, 
        imageResourceServerAddr, 
        serverPort
    );

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        terminal: false,
    });
    
    winston.info("Starting MCP server...");

    rl.on('line', async (line) => {
        try {
            const request = JSON.parse(line);
            // Assuming a simple JSON-RPC like structure: { "id": number, "method": string, "params": any }
            const { id, method, params } = request;
            let result;

            switch (method) {
                case 'get_info':
                    result = service.getInfo();
                    break;
                case 'generate_image':
                    result = await service.generate_image(params);
                    break;
                default:
                    throw new Error(`Unknown method: ${method}`);
            }
            // Write the JSON response back to stdout
            process.stdout.write(JSON.stringify({ id, result }) + '\n');

        } catch (e: any) {
            winston.error(`Error processing MCP request: ${e.message}`, { request_line: line });
            // Respond with an error
            const request = JSON.parse(line);
            process.stdout.write(JSON.stringify({ id: request.id, error: { message: e.message } }) + '\n');
        }
    });

    rl.on('close', () => {
        winston.info("MCP server stdin closed, shutting down.");
        httpServer.close(() => {
            winston.info("HTTP server shut down.");
            process.exit(0);
        });
    });
}

main().catch(err => {
    winston.error("Unhandled error in main function:", err);
    process.exit(1);
});
