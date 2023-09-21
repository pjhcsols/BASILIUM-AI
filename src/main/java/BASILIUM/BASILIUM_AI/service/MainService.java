package BASILIUM.BASILIUM_AI.service;

import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.util.pattern.PathPattern;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

public class MainService {
    private final String UPLOAD_DIR = "/home/woo/Desktop/job/project/VITON/BASILIUM-AI/model/dataset";
    String pythonFilePath = "/home/woo/Desktop/job/project/VITON/BASILIUM-AI/model/test.py";
    public Path aiModelService(String userId, String productId, MultipartFile [] files){
//        String fileName1 = userId + "_" + productId + "_" + "person" + ".png";
        String fileName1 = "/test_img/0.jpg";
//        String fileName2 = userId + "_" + productId + "_" + "product" + ".png";
        String fileName2 = "/test_clothes/0.jpg";
        Path path;
        try {
            File newFile1 = new File(UPLOAD_DIR + fileName1);
            File newFile2 = new File(UPLOAD_DIR + fileName2);
            files[0].transferTo(newFile1);
            files[1].transferTo(newFile2);

            // AI 모델을 호출하고 결과 파일의 경로를 반환
            path = callAiModel();
        } catch (IOException e) {
            throw new RuntimeException("파일 업로드 실패: " + e.getMessage(), e);
        }
        return path;
    }
    public Resource loadFile (Path filePath){
        try{
            return new UrlResource(filePath.toUri());
        } catch (MalformedURLException e){
            throw new RuntimeException("Malformed URL: " + filePath.toString(), e);
        }
    }

    public Path callAiModel(){
        System.out.println("Call AI Model");
        ProcessBuilder builder = new ProcessBuilder("python",pythonFilePath);
        builder.redirectErrorStream(true);
        Process process;

        try {
            process = builder.start();

        }catch (IOException e){
            return null;
        }
        try {
            process.waitFor(60, TimeUnit.SECONDS);
        }catch (InterruptedException e){
            return null;
        }
        process.destroy();
        String aiResultFilePath = "/home/woo/Desktop/job/project/VITON/BASILIUM-AI/model/results/0.jpg"; // 예시 결과 파일 경로
        return Path.of(aiResultFilePath);
    }
}
