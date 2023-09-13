package BASILIUM.BASILIUM_AI.controller;

import BASILIUM.BASILIUM_AI.service.MainService;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StreamUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;

@RestController
@RequestMapping("/basiliumAi")
public class MainController {

    private final ObjectMapper objectMapper;
    private final MainService mainService;

    @Autowired
    public MainController(ObjectMapper objectMapper, MainService mainService) {
        this.objectMapper = objectMapper;
        this.mainService = mainService;
    }

    @GetMapping("/getAiService")
    public ResponseEntity<byte[]> uploadProduct(@RequestParam String strInfos, @RequestParam("files")MultipartFile[] files){
        Path path;
        try{
            JsonNode rootNode = objectMapper.readTree(strInfos);
            String userId =rootNode.get("userId").asText();
            String productId = rootNode.get("productId").asText();
            path = mainService.aiModelService(userId, productId, files);
        } catch (Exception e){
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(null);
        }
        Resource imageResource = mainService.loadFile(path);
        try {
            InputStream inputStream = imageResource.getInputStream();
            byte[] imageBytes = StreamUtils.copyToByteArray(inputStream);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.IMAGE_PNG); // 이미지 타입에 맞게 설정

            return new ResponseEntity<>(imageBytes, headers, HttpStatus.OK);
        } catch (IOException e) {
            e.printStackTrace();
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

}
