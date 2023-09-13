package BASILIUM.BASILIUM_AI.service;

import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class MainService {
    private String UPLOAD_DIR = "C:/Users/kimmo/Desktop/img/";
    public Path aiModelService(String userId, String productId, MultipartFile [] files){
        String fileName1 = userId + "_" + productId + "_" + "person" + ".png";
        String fileName2 = userId + "_" + productId + "_" + "product" + ".png";

        try {
            File newFile1 = new File(UPLOAD_DIR + fileName1);
            File newFile2 = new File(UPLOAD_DIR + fileName2);
            files[0].transferTo(newFile1);
            files[1].transferTo(newFile2);

            // AI 모델을 호출하고 결과 파일의 경로를 반환
            Path aiResultFilePath = callAiModel(newFile1.getAbsolutePath(), newFile2.getAbsolutePath());
            return aiResultFilePath;
        } catch (IOException e) {
            throw new RuntimeException("파일 업로드 실패: " + e.getMessage(), e);
        }
    }
    public Resource loadFile (Path filePath){
        try{
            return new UrlResource(filePath.toUri());
        } catch (MalformedURLException e){
            throw new RuntimeException("Malformed URL: " + filePath.toString(), e);
        }
    }

    public Path callAiModel(String filePath1, String filePath2){
        // 여기에서 AI 모델을 호출하고 결과 파일의 경로를 반환하는 로직을 구현
        // filePath1과 filePath2는 업로드된 파일의 경로
        // 결과 파일의 경로를 Path 형식으로 반환
        String aiResultFilePath = "C:/path/to/ai/result.png"; // 예시 결과 파일 경로
        return Paths.get(aiResultFilePath);
    }
}
