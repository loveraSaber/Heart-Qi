class STAI:
    def __init__(self):
        self.state_anxiety = None #状态焦虑
        self.trait_anxiety = None #特质焦虑
        self.state_level = None #状态焦虑等级
        self.trait_level = None #特质焦虑等级
    
    def set_data(self, state_anxiety, trait_anxiety):
        self.state_anxiety = state_anxiety
        self.trait_anxiety = trait_anxiety

    def to_dict(self, include_none=True):
        """
        将类属性转换为字典
        
        Args:
            include_none: 是否包含值为None的属性
        """
        attributes = vars(self)
        
        if include_none:
            return attributes.copy()
        else:
            # 过滤掉值为None的属性
            return {key: value for key, value in attributes.items() 
                   if value is not None}
                   
    @classmethod
    def evaluate_anxiety_levels(cls, state_anxiety_score, trait_anxiety_score):
        """
        根据状态焦虑和特质焦虑分数评估焦虑水平
        
        参数:
        state_anxiety_score -- 状态焦虑分数(1-20项总分)
        trait_anxiety_score -- 特质焦虑分数(21-40项总分)
        
        返回:
        包含状态焦虑和特质焦虑水平的字典
        """

        stai = STAI()
        stai.set_data(state_anxiety_score, trait_anxiety_score)

        # 评估状态焦虑水平
        if stai.state_anxiety_score < 33:
            stai.state_level = "低水平状态焦虑"
        elif 33 <= stai.state_anxiety_score < 57:
            stai.state_level = "中等水平状态焦虑"
        else:
            stai.state_level = "高水平状态焦虑"
        
        # 评估特质焦虑水平
        if stai.trait_anxiety_score < 34:
            stai.trait_level = "低特质焦虑"
        elif 34 <= stai.trait_anxiety_score < 52:
            stai.trait_level = "中等特质焦虑"
        else:
            stai.trait_level = "高特质焦虑"
        
        return stai

        

class PHQ_9:
    def __init__(self):
        self.depression = None #抑郁
        self.depression_level = None #抑郁等级
    
    def set_data(self, depression):
        self.depression = depression

    @classmethod
    def evaluate_depression_levels(cls,depression_score):
        phq = PHQ_9()
        phq.set_data(depression_score)
        
        if phq.depression_score <= 13:
            phq.depression_level = "没有抑郁"
        elif phq.depression_score <= 18:
            phq.depression_level = "轻微抑郁"
        elif depression_score <= 23:
            phq.depression_level = "中度抑郁"
        elif phq.depression_score <= 28:
            phq.depression_level = "中重度抑郁"
        elif phq.depression_score <= 36:
            phq.depression_level = "重度抑郁"
        else:
            phq.depression_level = "超出范围"  # 处理分数大于36或小于0的情况（可根据需要调整）
        return phq

class CPSS_14:
    def __init__(self):
        self.pressure = None #感知压力
        self.pressure_level = None #感知压力等级
    
    def set_data(self, pressure):
        self.pressure = pressure
    
    @classmethod
    def evaluate_pressure_levels(cls, pressure_score):
        cpss = CPSS_14()
        cpss.set_data(pressure_score)

        if cpss.pressure_score <= 28:
            cpss.pressure_level = "感知压力水平较低"
        elif cpss.pressure_score <= 42:
            cpss.pressure_level = "感知压力水平适中"
        elif cpss.pressure_score <= 70:
            cpss.pressure_level = "感知压力水平高"
        return cpss
    
class AIS:
    def __init__(self):
        self.sleep_disorder = None  # 睡眠障碍
        self.sleep_disorder_level = None #睡眠障碍等级
        self.sleep_time = None #睡眠时间
        self.sleep_time_level = None #睡眠时间等级
        self.sleep_quality = None #睡眠质量
        self.sleep_quality_level = None #睡眠质量等级
        self.daytime_dysfunction = None #日间功能障碍
        self.daytime_dysfunction_level = None #日间功能障碍等级
    
    def set_data(self, sleep_disorder, sleep_time, sleep_quality, daytime_dysfunction):
        self.sleep_disorder = sleep_disorder
        self.sleep_time = sleep_time
        self.sleep_quality =sleep_quality
        self.daytime_dysfunction = daytime_dysfunction

    @classmethod
    def evluate_ais_levels(cls, sd_score, st_score, sq_score, dys_score):

        ais = AIS()
        ais.set_data(sd_score, st_score, sq_score, dys_score)
        
        if ais.sd_score<12:
            ais.sleep_disorder_level = "无睡眠障碍"
        elif ais.sd_score<14:
            ais.sleep_disorder_level = "存在失眠障碍风险"
        else:
            ais.sleep_disorder_level = "存在睡眠障碍"
        
        if ais.st_score <=1:
            ais.sleep_time_level = "睡眠时间充足"
        elif ais.st_score < 3:
            ais.sleep_time_level = "睡眠时间不足"
        else:
            ais.sleep_time_level = "睡眠时间严重不足"

        if ais.sq_score <=1:
            ais.sleep_quality_level = "睡眠质量良好"
        elif ais.sq_score < 3:
            ais.sleep_quality_level = "睡眠质量一般或较差"
        else:
            ais.sleep_quality_level = "睡眠质量差"
        
        if ais.dys_score <= 3:
            ais.daytime_dysfunction_level = "日间功能良好"
        elif ais.dys_score <6:
            ais.daytime_dysfunction_level = "存在日间功能障碍风险"
        else:
            ais.daytime_dysfunction_level == "存在日间功能障碍"
        return ais

class TFEQ_R18:
    def __init__(self):
        self.cog_restrictive_eating = None #认知限制性饮食
        self.cog_restrictive_eating_level = None #认知限制性饮食等级
        self.non_controlling_eating = None #非控制性饮食
        self.non_controlling_eating_level = None #非控制性饮食等级
        self.emotional_eating = None #情绪性饮食
        self.emotional_eating_level = None #情绪性饮食等级
        self.eating_behavior = None #进食行为
        self.eating_behavior_level = None #进食行为等级 

    def set_data(self, cog_restrictive_eating, non_controlling_eating, emotional_eating, eating_behavior):
        self.cog_restrictive_eating = cog_restrictive_eating
        self.non_controlling_eating = non_controlling_eating
        self.emotional_eating = emotional_eating
        self.eating_behavior = eating_behavior

    @classmethod
    def evalute_tfeq_levels(cls, cre, nce, emo, eb):

        tfeq = TFEQ_R18()
        tfeq.set_data(cre, nce, emo, eb)
        
        if tfeq.cog_restrictive_eating < 10:
            tfeq.cog_restrictive_eating_level = "没有饮食约束"
        elif tfeq.cog_restrictive_eating < 15:
            tfeq.cog_restrictive_eating_level = '低程度的饮食约束'
        elif tfeq.cog_restrictive_eating < 20:
            tfeq.cog_restrictive_eating_level = "限制性的饮食约束"
        else:
            tfeq.cog_restrictive_eating_level = "过度限制饮食约束"
        
        if tfeq.non_controlling_eating < 12:
            tfeq.non_controlling_eating_level = "进食自控能力强"
        elif tfeq.non_controlling_eating < 17:
            tfeq.non_controlling_eating_level = "进食自控能力一般"
        elif tfeq.non_controlling_eating < 22:
            tfeq.non_controlling_eating_level = "进食自控能力较差"
        else:
            tfeq.non_controlling_eating_level = "缺乏进食自控能力"
        
        if tfeq.emotional_eating < 4:
            tfeq.emotional_eating_level = '理性进食'
        elif tfeq.emotional_eating < 6:
            tfeq.eating_behavior_level = '潜在的情绪性进食风险'
        else:
            tfeq.eating_behavior_level = '存在情绪性进食'
        
        if tfeq.eating_behavior < 29:
            tfeq.emotional_eating_level = "进食行为良好"
        elif tfeq.eating_behavior < 50:
            tfeq.eating_behavior_level = "具有潜在的进食行为风险"
        else:
            tfeq.eating_behavior_level = "存在进食行为困扰"
        return tfeq

class BFI_2:
    def __init__(self):
        self.extroversion = None #外向性
        self.social = None #社交
        self.decisive = None #果断
        self.vitality = None #活力
        self.comfortable_nature = None #宜人性
        self.sympathy = None #同情
        self.humble = None #谦恭
        self.trust = None #信任
        self.responsibility = None #尽责性
        self.orderliness = None #条理
        self.efficiency = None #效率
        self.responsible = None #负责
        self.negative_emotions = None #神经质
        self.anxiety = None #焦虑
        self.depression = None #抑郁
        self.changeable = None #易变
        self.openness = None #开放性
        self.curious = None #好奇
        self.asethetic = None #审美
        self.imagine = None #想象
        self.extroversion_level = None  # 外向性等级
        self.social_level = None  # 社交等级
        self.decisive_level = None  # 果断等级
        self.vitality_level = None  # 活力等级
        self.comfortable_nature_level = None  # 宜人性等级
        self.sympathy_level = None  # 同情等级
        self.humble_level = None  # 谦恭等级
        self.trust_level = None  # 信任等级
        self.responsibility_level = None  # 尽责性等级
        self.orderliness_level = None  # 条理等级
        self.efficiency_level = None  # 效率等级
        self.responsible_level = None  # 负责等级
        self.negative_emotions_level = None  # 神经质等级
        self.anxiety_level = None  # 焦虑等级
        self.depression_level = None  # 抑郁等级
        self.changeable_level = None  # 易变等级
        self.openness_level = None  # 开放性等级
        self.curious_level = None  # 好奇等级
        self.asethetic_level = None  # 审美等级
        self.imagine_level = None  # 想象等级
    
    def set_data(self,
        extroversion,
        social,
        decisive,
        vitality,
        comfortable_nature,
        sympathy,
        humble,
        trust,
        responsibility,
        orderliness,
        efficiency,
        responsible,
        negative_emotions,
        anxiety,
        depression,
        changeable,
        openness,
        curious,
        asethetic,
        imagine
        ):
        self.extroversion = extroversion
        self.social = social
        self.decisive = decisive
        self.vitality = vitality
        self.comfortable_nature = comfortable_nature
        self.sympathy = sympathy
        self.humble = humble
        self.responsibility = responsibility
        self.orderliness = orderliness
        self.efficiency = efficiency
        self.responsible = responsible
        self.negative_emotions = negative_emotions
        self.anxiety = anxiety
        self.depression = depression
        self.changeable = changeable
        self.openness = openness
        self.curious = curious
        self.asethetic = asethetic
        self.imagine = imagine
    
    @classmethod
    def evalutate_bfi_levels(cls,
        extroversion,
        social,
        decisive,
        vitality,
        comfortable_nature,
        sympathy,
        humble,
        trust,
        responsibility,
        orderliness,
        efficiency,
        responsible,
        negative_emotions,
        anxiety,
        depression,
        changeable,
        openness,
        curious,
        asethetic,
        imagine
        ):
        bfi = BFI_2()
        bfi.set_data(
        extroversion,
        social,
        decisive,
        vitality,
        comfortable_nature,
        sympathy,
        humble,
        trust,
        responsibility,
        orderliness,
        efficiency,
        responsible,
        negative_emotions,
        anxiety,
        depression,
        changeable,
        openness,
        curious,
        asethetic,
        imagine
        )
        if bfi.extroversion < 34:
            bfi.extroversion_level = "低外向性"
        elif bfi.extroversion < 46:
            bfi.extroversion_level = "中等外向性"
        else:
            bfi.extroversion_level = "高外向性"
        
        if bfi.social < 9:
            bfi.social_level = "低社交"
        elif bfi.social < 16:
            bfi.social_level = "中等社交"
        else:
            bfi.social = "高社交"
        
        if bfi.decisive < 10:
            bfi.decisive_level = "低果断"
        elif bfi.decisive < 16:
            bfi.decisive_level = "中等果断"
        else: 
            bfi.decisive_level = "高果断"
        
        if bfi.vitality < 12:
            bfi.vitality_level = "低活力"
        elif bfi.vitality < 17:
            bfi.vitality_level = "中等活力"
        else:
            bfi.vitality_level = "高活力"
        
        if bfi.comfortable_nature < 41:
            bfi.comfortable_nature_level = "低宜人性"
        elif bfi.comfortable_nature < 53:
            bfi.comfortable_nature_level = "中等宜人性"
        else:
            bfi.comfortable_nature_level = "高宜人性"
        
        if bfi.sympathy < 14:
            bfi.sympathy_level = "低同情"
        elif bfi.sympathy < 18:
            bfi.sympathy_level = "中等同情"
        else:
            bfi.sympathy_level = "高同情"
        
        if bfi.humble < 13:
            bfi.humble_level = "低谦恭"
        elif bfi.humble < 18:
            bfi.humble_level = "中等谦恭"
        else: 
            bfi.humble_level = "高谦恭"
        
        if bfi.trust < 13:
            bfi.trust_level = "低信任"
        elif bfi.trust<=18:
            bfi.trust_level = "中等信任"
        else:
            bfi.trust_level= "高信任"


        if bfi.responsibility < 38:
            bfi.responsibility_level = "低尽责性"
        elif bfi.responsibility < 52:
            bfi.responsibility_level = "中等尽责性"
        else:
            bfi.responsibility_level = "高尽责性"
        
        
        if bfi.orderliness < 12:
            bfi.orderliness_level = "低条理"
        elif bfi.orderliness < 18:
            bfi.orderliness_level = "中等条理"
        else:
            bfi.orderliness_level = "高条理"

        
        if bfi.efficiency < 12:
            bfi.efficiency_level = "低效率"
        elif bfi.efficiency < 18:
            bfi.efficiency_level = "中等效率"
        else:
            bfi.efficiency_level = "高效率"
        
        if bfi.responsible < 12:
            bfi.responsible_level = "低负责"
        elif bfi.responsible < 18:
            bfi.responsible = "中等负责"
        else: 
            bfi.responsible = "高负责"
        
        if bfi.negative_emotions < 23:
            bfi.negative_emotions_level = "低神经质"
        elif bfi.negative_emotions < 38:
            bfi.negative_emotions_level = "中等神经质"
        else:
            bfi.negative_emotions_level = "高神经质"
        
        if bfi.anxiety < 9:
            bfi.anxiety_level = "低焦虑"
        elif bfi.anxiety < 14:
            bfi.anxiety_level = "中等焦虑"
        else:
            bfi.anxiety_level = "高焦虑"
        
        if bfi.depression < 6:
            bfi.depression_level = "低抑郁"
        elif bfi.depression < 12:
            bfi.depression_level = "中等抑郁"
        else:
            bfi.depression_level = "高抑郁"
        
        if bfi.changeable < 6:
            bfi.changeable_level = "低易变"
        elif bfi.changeable < 12:
            bfi.changeable_level = "中等易变"
        else:
            bfi.changeable_level = "高易变"
        
        if bfi.openness < 33:
            bfi.openness_level = "低开放性"
        elif bfi.openness < 46:
            bfi.openness_level = "中等开放性"
        else:
            bfi.openness_level = "高开放性"
        
        if bfi.curious < 11:
            bfi.curious_level = "低好奇"
        elif bfi.curious < 15:
            bfi.curious_level = "中等好奇"
        else:
            bfi.curious_level = "高好奇"
        
        if bfi.asethetic < 9:
            bfi.asethetic_level = "低审美"
        elif bfi.asethetic < 16:
            bfi.asethetic_level = "中等审美"
        else:
            bfi.asethetic_level = "高审美"
        
        if bfi.imagine < 11:
            bfi.imagine_level = "低想象"
        elif bfi.imagine < 17:
            bfi.imagine_level = "中等想象"
        else:
            bfi.imagine_level = "高想象"
        return bfi

    
class FFMQ:
    def __init__(self):
        self.observation = None  # 观察 - 对内外在体验的觉察能力
        self.observation_level = None  # 观察等级
        self.description = None  # 描述 - 用语言表达观察内容的能力
        self.description_level = None  # 描述等级
        self.act_wareness = None  # 觉知的行动 - 在有意识觉察下行动的能力
        self.act_wareness_level = None  # 觉知的行动等级
        self.no_judge = None  # 不判断 - 对体验不做评价的接纳态度
        self.no_judge_level = None  # 不判断等级
        self.no_action = None  # 不行动 - 保持静止和观察而不反应的能力
        self.no_action_level = None  # 不行动等级
    
    def set_data(self, observation, description, act_wareness, no_judge, no_action):
        self.observation = observation
        self.description = description
        self.act_wareness = act_wareness
        self.no_judge = no_judge
        self.no_action = no_action

    @classmethod
    def evalutate_ffmq_levels(cls, observation, description, act_wareness, no_judge, no_action):

        ffmq = FFMQ()
        ffmq.set_data(observation, description, act_wareness, no_judge, no_action)
        if ffmq.observation < 15:
            ffmq.observation_level = "较低的正念观察水平"
        elif ffmq.observation <= 25:
            ffmq.observation_level = "中等正念观察水平"
        else:
            ffmq.observation_level = "较高的正念观察水平"
    
        if ffmq.description < 18:
            ffmq.description_level = "较低的正念描述水平"
        elif ffmq.description < 28:
            ffmq.description_level = "中等正念描述水平"
        else:
            ffmq.description_level = "较高的正念描述水平"
        
        if ffmq.act_wareness < 20:
            ffmq.act_wareness_level = "较低水平的觉知地行动"
        elif ffmq.act_wareness < 30:
            ffmq.act_wareness_level = "中等水平的觉知地行动"
        else:
            ffmq.act_wareness_level = "较高水平的觉知地行动"

        if ffmq.no_judge < 15:
            ffmq.no_judge_level = "较低程度的不判断"
        elif ffmq.no_judge < 25:
            ffmq.no_judge_level = "中等程度的不判断"
        else:
            ffmq.no_judge_level = "较高程度的不判断"

        if ffmq.no_action < 12:
            ffmq.no_action_level = "较低程度的不行动"
        elif ffmq.no_action < 25:
            ffmq.no_action_level = "中等程度的不行动"
        else:
            ffmq.no_action_level = "较高程度的不行动"

        return ffmq


class BPS:
    def __init__(self):
        self.borderline_personality_disorder = None  # 边缘性人格障碍 - 边缘性人格特质表现程度
        self.borderline_personality_disorder_level = None  # 边缘性人格障碍等级

        self.emotional_stability = None  # 情绪稳定性 - 情绪波动的控制能力
        self.emotional_stability_level = None  # 情绪稳定性等级

        self.self_awareness = None  # 自我认知 - 对自我状态、动机和情感的了解程度
        self.self_awareness_level = None  # 自我认知等级

        self.interpersonal_function = None  # 人际功能 - 建立和维护人际关系的能力
        self.interpersonal_function_level = None  # 人际功能等级

        self.regulation_strategy = None  # 调节策略 - 情绪和行为调节技巧的掌握程度
        self.regulation_strategy_level = None  # 调节策略等级
    
    def set_data(self, borderline_personality_disorder, emotional_stability, self_awareness, interpersonal_function, regulation_strategy):
        self.borderline_personality_disorder = borderline_personality_disorder
        self.emotional_stability = emotional_stability
        self.self_awareness = self_awareness
        self.interpersonal_function = interpersonal_function
        self.regulation_strategy = regulation_strategy
    
    @classmethod
    def evalute_bps_levels(cls, bpd, emo, sa, ifun ,rs):
        bps = BPS()
        bps.set_data(bpd, emo, sa, ifun, rs)
        
        if bps.borderline_personality_disorder < 18:
            bps.borderline_personality_disorder_level = "不存在边缘性人格障碍风险"
        elif bps.borderline_personality_disorder < 34:
            bps.borderline_personality_disorder_level = "存在较低的边缘性人格障碍风险"
        elif bps.borderline_personality_disorder:
            bps.borderline_personality_disorder_level ="存在边缘性人格障碍风险"

        if bps.emotional_stability < 4:
            bps.emotional_stability_level = "情绪稳定"
        elif bps.emotional_stability < 8:
            bps.emotional_stability_level = "情绪比较稳定"
        else:
            bps.emotional_stability_level = "情绪不稳定"
        
        if bps.self_awareness < 4:
            bps.self_awareness_level = "自我认知功能较好"
        elif bps.self_awareness < 8:
            bps.self_awareness_level = "自我认知功能正常"
        else:
            bps.self_awareness_level = "自我认知功能较差"
        
        if bps.interpersonal_function < 4:
            bps.interpersonal_function_level = "人际功能良好"
        elif bps.interpersonal_function<8:
            bps.borderline_personality_disorder_level = "人际功能一般"
        else:
            bps.borderline_personality_disorder_level = "人际功能较差"

        if bps.regulation_strategy < 4:
            bps.regulation_strategy_level = "调节策略良好"
        elif bps.regulation_strategy < 8:
            bps.regulation_strategy_level = "调节策略一般"
        else:
            bps.regulation_strategy_level = "调节策略较差"
        return bps
            
class PsychologicalAssessment:
    def __init__(self):
        # 把所有量表都作为属性
        self.stai = STAI()
        self.phq_9 = PHQ_9()
        self.cpss_14 = CPSS_14()
        self.ais = AIS()
        self.tfeq_r18 = TFEQ_R18()
        self.bfi_2 = BFI_2()
        self.ffmq = FFMQ()
        self.bps = BPS()

    def get_summary(self):
        """
        汇总结果，可以根据需要调整
        """
        return {
            "STAI": {
                "state_anxiety": self.stai.state_anxiety,
                "trait_anxiety": self.stai.trait_anxiety,
            },
            "PHQ-9": {
                "depression": self.phq_9.depression,
            },
            "CPSS-14": {
                "pressure": self.cpss_14.pressure,
            },
            "AIS": {
                "sleep_disorder": self.ais.sleep_disorder,
                "sleep_time": self.ais.sleep_time,
                "sleep_quality": self.ais.sleep_quality,
                "daytime_dysfunction": self.ais.daytime_dysfunction,
            },
            "TFEQ-R18": {
                "cog_restrictive_eating": self.tfeq_r18.cog_restrictive_eating,
                "non_controlling_eating": self.tfeq_r18.non_controlling_eating,
                "emotional_eating": self.tfeq_r18.emotional_eating,
                "eating_behavior": self.tfeq_r18.eating_behavior,
            },
            "BFI-2": {
                "extroversion": self.bfi_2.extroversion,
                "social": self.bfi_2.social,
                "decisive": self.bfi_2.decisive,
                "vitality": self.bfi_2.vitality,
                "comfortable_nature": self.bfi_2.comfortable_nature,
                "sympathy": self.bfi_2.sympathy,
                "humble": self.bfi_2.humble,
                "responsibility": self.bfi_2.responsibility,
                "orderliness": self.bfi_2.orderliness,
                "efficiency": self.bfi_2.efficiency,
                "responsible": self.bfi_2.responsible,
                "negative_emotions": self.bfi_2.negative_emotions,
                "anxiety": self.bfi_2.anxiety,
                "depression": self.bfi_2.depression,
                "changeable": self.bfi_2.changeable,
                "openness": self.bfi_2.openness,
                "curious": self.bfi_2.curious,
                "asethetic": self.bfi_2.asethetic,
                "imagine": self.bfi_2.imagine,
            },
            "FFMQ": {
                "observation": self.ffmq.observation,
                "description": self.ffmq.description,
                "act_wareness": self.ffmq.act_wareness,
                "no_judge": self.ffmq.no_judge,
                "no_action": self.ffmq.no_action,
            },
            "BPS": {
                "borderline_personality_disorder": self.bps.borderline_personality_disorder,
                "emotional_stability": self.bps.emotional_stability,
                "self_awareness": self.bps.self_awareness,
                "interpersonal_function": self.bps.interpersonal_function,
                "regulation_strategy": self.bps.regulation_strategy,
            }
        }
    































































