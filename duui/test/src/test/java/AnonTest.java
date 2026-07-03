import org.apache.uima.UIMAException;
import org.apache.uima.cas.SerialFormat;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.apache.uima.fit.factory.JCasFactory;
import org.texttechnologylab.annotation.type.Image;
import org.xml.sax.SAXException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.UnknownHostException;
import java.util.Base64;
import java.util.List;

public class AnonTest {

    static DUUIComposer composer;
    static JCas cas;

//    static String url = "http://isengart.hucompute.org:9991";
    static String url = "http://127.0.0.1:8001";
    static String inputPath = "resources/input";
    static String sOutputPath = "resources/output";

    @BeforeAll
    static void beforeAll() throws Exception {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIUIMADriver uima_driver = new DUUIUIMADriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver, uima_driver);
        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        composer.addDriver(docker_driver);

        cas = JCasFactory.createJCas();
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }

    @AfterEach
    public void afterEach() throws Exception {
        composer.resetPipeline();

        readImagesInCas();
        cas.reset();
    }


    // Helper method to convert image file to Base64 string
    private static List<String> convertImageToBase64(String imagePath) {
        try {
            File file = new File(imagePath);
            BufferedImage bufferedImage = ImageIO.read(file);
            int height = bufferedImage.getHeight();
            int width = bufferedImage.getWidth();

            // Convert the image to byte array
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            ImageIO.write(bufferedImage, "png", byteArrayOutputStream);
            byte[] imageBytes = byteArrayOutputStream.toByteArray();

            // Encode the byte array to Base64

            return List.of(Base64.getEncoder().encodeToString(imageBytes), Integer.toString(height), Integer.toString(width));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
    // Helper method to save Base64 string back to an image file
    private static void saveBase64ToImage(String base64String, String name) {
        try {
            // Decode the Base64 string into a byte array
            byte[] decodedBytes = Base64.getDecoder().decode(base64String);

            // Create an image from the byte array
            InputStream inputStream = new ByteArrayInputStream(decodedBytes);
            BufferedImage image = ImageIO.read(inputStream);

            // Save the image to the specified output file
            File outputFile = new File(sOutputPath + "/" +name + ".png");
            ImageIO.write(image, "png", outputFile);

            System.out.println("Image saved as: " + sOutputPath+name);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void createCas() throws UIMAException{
        cas.setDocumentLanguage("eng");
        cas.setDocumentText("Place holder text");
        File inputFolder = new File(inputPath);
        System.out.println("Reading in images from: " + inputFolder);
        for (File file: inputFolder.listFiles()){
            if (! file.isDirectory()){
                Image image = new Image(cas);
                List<String> imageInfo = convertImageToBase64(file.getPath());
                System.out.println(imageInfo.get(1));
                System.out.println(imageInfo.get(2));
                image.setSrc(imageInfo.get(0));
                image.setMimetype("img/");
                image.setHeight(Integer.parseInt(imageInfo.get(1)));
                image.setWidth(Integer.parseInt(imageInfo.get(2)));

                image.addToIndexes();
            }
        }
    }

    public void readImagesInCas() {
        int i = 0;
        for (Image image : JCasUtil.select(cas, Image.class)){
            saveBase64ToImage(image.getSrc(), String.valueOf(i));
            String result = "Image %d is has specified: \n Height: %d; Width: %d".formatted(i, image.getHeight(), image.getWidth());
            System.out.println(result);
        }

    }

    /*
    to visualize the cas to check if it was initialized correctly
     */
    public static void simpleTempExporter(JCas cas, String filename) throws IOException{
        File outputFile = new File(sOutputPath + "/"+ filename + ".xmi");
        outputFile.getParentFile().mkdirs();
        CasIOUtils.save(cas.getCas(), new FileOutputStream(outputFile), SerialFormat.XMI_PRETTY);
    }

    @Test
    public void testSingleFaceSimple() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "single_align")
                        .withParameter("vis_input", "true")
                        .build().withTimeout(1000)

        );

        createCas();
        simpleTempExporter(cas, "testSingleFaceSimple");
        composer.run(cas);
    }
}
