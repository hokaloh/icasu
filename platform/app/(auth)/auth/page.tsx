import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState } from "react";
import { useRouter } from "next/router";

type Tab = "sign-in" | "sign-up";

export function TabList() {
    return (
      <TabsList>
        <TabsTrigger value="sign-in">Sign In</TabsTrigger>
        <TabsTrigger value="sign-up">Sign Up</TabsTrigger>
      </TabsList>
    )
}

export default function Authentication(){    
    const router = useRouter();
    const [selectedTab, setSelectedTab] = useState<Tab>("sign-in");
    



    return (<Tabs>
    
    </Tabs>)
} 