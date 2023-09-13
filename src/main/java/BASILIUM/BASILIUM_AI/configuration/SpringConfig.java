package BASILIUM.BASILIUM_AI.configuration;

import BASILIUM.BASILIUM_AI.service.MainService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SpringConfig {

    public SpringConfig() {
    }

    @Bean
    public MainService mainService(){
        return new MainService();
    }
}
