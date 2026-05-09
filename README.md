# MiaoInsight
一個幫助公廟分析人流和做決策的系統

#專案想法與筆記
廟方 : 
會舉辦活動，分成定期舉辦的歲時祭儀，不定期舉辦的某些活動(可能是臨時起意或是突然有合作活動之類的)。
有歷年活動的報名者基本資料記錄。
想改善的重點是行銷(數位化、國際聲譽、成本改善、....)，但我們發現管理上他們也有改善空間，並且廟方有提到，只要可以幫助他們改善的東西他們都要。

我們想做的專案:
用廟方有記錄的資料建立基本模型，例如時間活動的時間資料可以拿來切很細，分析辦某個活動時人流變化；報名者的資料(會友報名的活動...)則可以分析每個活動預期有多少人參加。
且系統分成簡易模式與建議模式:
簡易模式 : 用選單選擇某場特定活動，系統會自動跑出可以調整的參數，廟方人員就可以知道在只改動一兩個小地方(如增設幾張桌子，將日期提前或延後，設置怎樣的攤販或是活動獎勵等等)對活動的人流以及收益會有怎樣的影響。
建議模式:嵌入大語言模型，當廟方有改動較大或是新的點子，就可以透過該模式透過敘述的方式描述他們具體想要怎麼做。系統會結合模型與其他資料(目前想到的是去爬新聞)，給出實際建議和預期的人流與收益。

version 3
我沒有分成大中小這類的 分成有無報名 因為大中小預測邏輯都一樣 之後有數據在微調係數就好
registration_based:{
event_type
event_category
days_before_event
registration_count
registration_growth_rate
ig_exposure
fb_exposure
threads_exposure
trend_velocity
reward_score
marketing_push_score
influencer_score
historical_show_rate
weather_score
holiday_score
duration_hours
venue_capacity
}

open:{
event_type
event_category
days_before_event
nearby_foot_traffic
foot_traffic_growth
ig_exposure
fb_exposure
threads_exposure
trend_velocity
reward_score
marketing_push_score
venue_visibility_score
weather_score
holiday_score
booth_count
live_performance_score
transport_accessibility
venue_capacity
}
輸出:人數、95%CI、forecast_event_timeline、feature importance、model score、可視圖
